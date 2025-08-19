import argparse
# import umbridge
import numpy as np
from collections import deque
import time

import clt_mlmcmc.classes.distributions as distributions
from clt_mlmcmc.data.process_data import forward_model
from clt_mlmcmc.classes.umbridge_server import CoupledTsunamiLandslide


class MLMCMCSolver():
    """
    Bayesian Inverse Problem Solution via MLMCMC, using Random Walk.
    Initialisation:
    model (CoupledTsunamiLandslide): the model to use for the forward
        simulation.
    multilevel_config (dict): contains the configuration for each level of the
        multilevel MCMC. Each key is a level, and each value is a dict with
        the following keys:
            level (int): the level of the MCMC
            samples (int): number of samples to take at this level
            time-step-relaxation (float): between 0 and 1, for global adaptive
                time steps
            min-depth (int): minimum depth for the simulation
            noise_std (float): standard deviation of the noise in the data
            proposal_std (float): standard deviation of the proposal
                distribution
    data (np.ndarray): observed data to use for the likelihood.
    unknown (str): name of the unknown variable in the forward model.
    """

    def __init__(self, model, multilevel_config, data=None, unknown='h1u'):
        self.data = data
        self.configs = multilevel_config
        self.system = [self._config_system(val, self.data, model, unknown)
                       for _, val in multilevel_config.items()]
        self.chains = [[] for _ in range(len(self.system))]
        self.acceptance = [[] for _ in range(len(self.system))]

    def _config_system(self, config, data, model, unknown):
        """
        Set up system for MLMCMC level.
        Input:
        config (dict): contains resolution parameters for solver and chain
        setup:
            level (int): the level of the MCMC
            samples (int): number of samples to take at this level
            time-step-relaxation (float): between 0 and 1, for global adaptive
                time steps
            min-depth (int): minimum depth for the simulation
            noise_std (float): standard deviation of the noise in the data
            proposal_std (float): standard deviation of the proposal
                distribution
        data (np.ndarray): observed data to use for the likelihood.
        model (CoupledTsunamiLandslide): the model to use for the forward.
        """
        config_items = ['level', 'samples', 'time-step-relaxation',
                        'min-depth', 'noise_std', 'proposal_std']
        for item in config_items:
            if item not in config.keys():
                raise ValueError(f"config needs {config_items}")
        return {
            # TODO: probably want to not hard code the use of distributions so
            # it's more general. but for now i am going to hard code it
            'prior': distributions.Uniform(lower=0.001, upper=0.01),
            'likelihood': distributions.Normal(
                mean=data,cov=config['noise_std']**2*np.eye(data.size)
            ),
            'proposal': distributions.Normal(
                mean=0, cov=config['proposal_std']**2
            ),
            'forward_model': lambda x, y: forward_model(x.tolist()[0], config,
                                                        model, unknown,
                                                        original_call=y),
            'level': config['level'],
            'samples': config['samples']
        }

    def generate_chain(self, level, start=None, subsampling=False):
        """
        Generate a chain for the given level.
        Input:
        level (int): the level of the MLMCMC to generate the chain for.
        start (float or None): the starting point for the chain. If None,
            the starting point will be sampled from the prior distribution.
        subsampling (bool): whether to use subsampling or not. If True, the
            chain will be generated using subsampling, otherwise it will be
            generated using coupled random walk.
        """
        if level not in range(len(self.system)):
            raise ValueError('level needs to be a non-negative integer '
                             f'smaller than {len(self.system)}.')
        system = self.system[level]
        if start is None:
            start = system['prior'].sample(1)[0]
        if not level:
        # if True:
            chain, acceptance = self._randomwalk(system, start,
                                                 subsampling=subsampling)
            self.chains[0] = chain
            self.acceptance[0] = acceptance
            return chain, acceptance
        else:
            if subsampling:
                coarse_n = self.system[level-1]['samples']
                fine_n = system['samples']
                if fine_n > coarse_n:
                    raise ValueError("Number of samples must be decreasing "
                                     "with increasing levels.")
                indexes = np.arange(0, len(self.posteriors[level-1]),
                                    coarse_n//fine_n)
                if not level-1:
                    if not len(self.chains[0]):
                        raise ValueError("Chain 0 has not been generated yet.")
                    coarse_chain = self.chains[0][indexes]
                else:
                    if not len(self.chains[level-1][0]):
                        raise ValueError(f"Chain {level-1} has not been "
                                         "generated yet.")
                    coarse_chain = self.chains[level-1][0][indexes]
                coarse_posterior = self.posteriors[level-1][indexes]
                chain_fine, chain_coarse, acceptances =\
                    self._subsampledrandomwalk(
                        system, coarse_chain, coarse_posterior, start
                    )
                self.chains[level] = [chain_fine, chain_coarse]
                self.acceptance[level] = acceptances
                return chain_fine, chain_coarse, acceptances
            else:
                system_fine = system
                system_coarse = self.system[level-1]
                chain_fine, chain_coarse, acceptances =\
                    self._coupledrandomwalk(
                        system_fine, system_coarse, start
                    )
                self.chains[level] = [chain_fine, chain_coarse]
                self.acceptance[level] = acceptances
                return chain_fine, chain_coarse, acceptances
        
    def generate_solution(self, start=None, subsampling=False):
        """
        Generate the solution for the MLMCMC.
        Input:
        start (float or None): the starting point for the chain. If None,
            the starting point will be sampled from the prior distribution.
        subsampling (bool): whether to use subsampling or not. If True, the
            chain will be generated using subsampling, otherwise it will be
            generated using coupled random walk. Used for multi-level chains.
        """
        self.solution = 0
        for i in range(len(self.chains)):
            print(f"Level {i}")
            self.generate_chain(i, start=start, subsampling=subsampling)
            if not i:
                self.solution += self.chains[0].mean()
            else:
                self.solution += self.chains[i][0].mean()\
                    - self.chains[i][1].mean()
        return self.chains, self.solution

    def _randomwalk(self, system, initial, subsampling=False):
        """
        Generate a random walk chain for the given system.
        Input:
        system (dict): the system to generate the chain for.
        initial (float): the initial value for the chain.
        subsampling (bool): whether to use subsampling or not. If True, the
            chain will be generated using subsampling, otherwise it will be
            generated using a simple random walk. Used for multi-level chains.
        """
        n = system['samples']-1
        steps = system['proposal'].sample(size=n).flatten()
        rng = np.log(np.random.uniform(0, 1, size=n))
        markov_chain = [np.array(initial)]
        current = initial
        posterior = lambda x: system['prior'].logpdf(x) +\
            system['likelihood'].logpdf(system['forward_model'](
                np.array([x]), False
            ))
        current_posterior = system['prior'].logpdf(current) +\
            system['likelihood'].logpdf(system['forward_model'](
                np.array([current]), True
            ))
        acceptance = 1
        if subsampling:
            posteriors = [current_posterior]
        for i in range(n):
            if not (i+1) % 50:
                print(f"Sample {i+1}/{n+1}")
            proposal = current + steps[i]
            print(proposal)
            if not np.isneginf(system['prior'].logpdf(proposal)):
                proposal_posterior = posterior(proposal)
                if rng[i] < proposal_posterior - current_posterior:
                    current = proposal
                    current_posterior = proposal_posterior
                    acceptance += 1
            markov_chain.append(current)
            if subsampling:
                posteriors.append(current_posterior)
        # in the subsampling case, only using this function for level 0
        if subsampling:
            self.posteriors = [np.array(posteriors)]
        return np.array(markov_chain).flatten(), acceptance/(n+1)

    def _coupledrandomwalk(self, system_fine, system_coarse, initial):
        """
        Generate a pair of coupled random walk chains for the given system,
        where both chains step the same amount at each step.
        Input:
        system (dict): the system to generate the (fine) chain for.
        initial (float): the initial value for the (fine) chain.
        subsampling (bool): whether to use subsampling or not. If True, the
            chain will be generated using subsampling, otherwise it will be
            generated using a simple random walk. Used for multi-level chains.
        """
        n = system_fine['samples'] - 1
        steps = system_fine['proposal'].sample(size=n).flatten()
        rng = np.log(np.random.uniform(0, 1, size=n))
        # [[fine], [coarse]]
        chain = [[initial], [initial]]
        current = np.array([initial, initial])
        posterior = lambda x: np.array([
            system_fine['prior'].logpdf(x[0]) +
            system_fine['likelihood'].logpdf(
                system_fine['forward_model'](np.array([x[0]]), False)
            ),
            system_coarse['prior'].logpdf(x[1]) +
            system_coarse['likelihood'].logpdf(
                system_coarse['forward_model'](np.asarray([x[1]]), False)
            )
        ])
        current_posterior = np.array([
            system_fine['prior'].logpdf(current[0]) +
            system_fine['likelihood'].logpdf(
                system_fine['forward_model'].original_call(
                    np.array([current[0]]), True
                )
            ),
            system_coarse['prior'].logpdf(current[1]) +
            system_coarse['likelihood'].logpdf(
                system_coarse['forward_model'].original_call(
                    np.asarray([current[1]]), True
                )
            )
        ])
        acceptance = [0, 0]
        for i in range(n):
            if not (i+1) % 50:
                print(f"Sample {i+1}/{n+1}")
            proposal = current + steps[i]
            proposal_posterior = posterior(proposal)
            difference = proposal_posterior - current_posterior
            for j in range(2):
                if rng[i] < difference[j]:
                    current[j] = proposal[j]
                    current_posterior[j] = proposal_posterior[j]
                    acceptance[j] += 1
                chain[j].append(current[j].copy())
        return [
            np.array(chain[0]).flatten(), # fine chain
            np.array(chain[1]).flatten(), # coarse chain
            np.array(acceptance)/n # acceptance rates for both chains
        ]

    def _subsampledrandomwalk(self, system, coarse_chain,
                              coarse_posterior, initial):
        """
        Generate a pair of coupled random walk chains for the given system,
        where the coarse chain is subsampled from the fine chain of the
        previous chain.
        Input:
        system (dict): the system to generate the (fine) chain for.
        coarse_chain (np.ndarray): the coarse chain from the previous level.
        coarse_posterior (np.ndarray): the posterior of the coarse chain from
            the previous level.
        initial (float): the initial value for the (fine) chain.
        """
        n = system['samples'] - 1
        steps = system['proposal'].sample(size=n).flatten()
        rng = np.log(np.random.uniform(0, 1, size=n))
        old_chain = deque(coarse_chain)
        old_posteriors = deque(coarse_posterior)
        old_initial = old_chain.popleft()
        finalised_coarse = [old_initial]
        finalised_new = [initial]
        current = [old_initial, initial]
        posterior = lambda x: system['prior'].logpdf(x) +\
            system['likelihood'].logpdf(
                system['forward_model'](np.array([x]), False)
            )
        current_posterior = system['prior'].logpdf(current[1]) +\
            system['likelihood'].logpdf(
                system['forward_model'](np.array([current[1]]), True)
            )
        current_correction = old_posteriors.popleft()
        acceptance = 1
        new_posteriors = [current_posterior]
        for i in range(n):
            if not (i+1) % 50:
                print(f"Sample {i+1}/{n+1}")
            proposal = current[1] + steps[i]
            proposal_coarse = old_chain.popleft()
            proposal_correction = old_posteriors.popleft()
            print(proposal)
            if not np.isneginf(system['prior'].logpdf(proposal)):
            # check if this is correct
                proposal_posterior = posterior(proposal)
                difference = proposal_posterior - proposal_correction -\
                    current_posterior + current_correction
                if rng[i] < difference:
                    current = [proposal_coarse, proposal]
                    current_posterior = proposal_posterior
                    current_correction = proposal_correction
                    acceptance += 1
            finalised_coarse.append(current[0])
            finalised_new.append(current[1])
            new_posteriors.append(current_posterior)
        self.posteriors.append(np.array(new_posteriors))
        return [
            np.array(finalised_new).flatten(),
            np.array(finalised_coarse).flatten(),
            acceptance/(n+1)
        ]

    def save_data(self, filename='MLMCMC_samples.csv'):
        """
        Save data from all chains into one csv file.
        """
        for level in range(len(self.chains)):
            if not level:
                if not len(self.chains[level]):
                    raise ValueError("Chain 0 has not been generated yet.")
                data = self.chains[0]
            else:
                if not (len(self.chains[level][0])
                        or len(self.chains[level][1])):
                    raise ValueError(f"Chain {level} has not been generated "
                                     "yet.")
                data = np.concatenate([data, self.chains[level][0],
                                    self.chains[level][1]])
        np.savetxt(filename, data, delimiter=',')
    
    def _ess(self, chain):
        """
        Effective Sample Size for a 1D chain.
        Input:
        chain (np.ndarray): the chain to calculate the ESS for.
        """
        x = np.asarray(chain)
        N = len(x)
        x_centered = x - np.mean(x)
        acf = np.correlate(x_centered, x_centered, mode='full')
        acf = acf[acf.size // 2:] / acf[acf.size // 2]  # normalize
        
        # Geyer's initial positive sequence: stop when sum of pairs < 0
        tau = 0.0
        for k in range(1, N):
            rho = acf[k]
            if rho < 0:
                break
            tau += rho
        ess_val = N / (1 + 2 * tau)
        return ess_val

    def effective_sample_size(self):
        """
        Calculate the effective sample size for each level of the MLMCMC.
        Returns:
        ess (list): a list of effective sample sizes for each level.
        """
        ess = []
        for i in range(len(self.system)):
            print(f"LEVEL {i}:")
            if i:
                fine = self._ess(self.chains[i][0])
                coarse = self._ess(self.chains[i][1])
                ess.append([fine, coarse])
            else:
                ess.append(self._ess(self.chains[i]))
        return ess

parser = argparse.ArgumentParser(description='Model output test.')
args = parser.parse_args()
model = CoupledTsunamiLandslide()
y_obs = np.loadtxt("data/observed_data_noise.csv", delimiter=',')

# Improvement Area: Automation of 'noise_std', 'proposal_std' and 'samples' for
# optimisation wrt computational resources.
multilevel_config = {
    0: {
        'level': 0,
        'samples': 1000,
        'time-step-relaxation': 0.7,
        'min-depth': 3,
        'noise_std': 3,
        'proposal_std': 0.00492
    },
    1: {
        'level': 1,
        'samples': 1000,
        'time-step-relaxation': 0.45,
        'min-depth': 3,
        'noise_std': 2,
        'proposal_std': 0.004
    },
    2: {
        'level': 2,
        'samples': 500,
        'time-step-relaxation': 0.45,
        'min-depth': 4,
        'noise_std': 0.8,
        'proposal_std': 0.0024
    }
}

start = time.time()
solver = MLMCMCSolver(model, multilevel_config, y_obs)
data, solution = solver.generate_solution(subsampling=True)
end = time.time()
solver.save_data()
solver.effective_sample_size()
print(solver.acceptance)
print(solution)
print(f"time: {end-start}")
