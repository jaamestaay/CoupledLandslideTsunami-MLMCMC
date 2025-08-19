
import os
import time
import umbridge

# More can be added if required for parameterisation of model.
config_items = ['time-step-relaxation', 'min-depth']

class CoupledTsunamiLandslide(umbridge.Model):
    def __init__(self):
        super().__init__("CoupledTsunamiLandslide")

    def get_input_sizes(self, config):
        return [1]  # Input is friction parameter

    def get_output_sizes(self, config):
        return [1]  # Max. eigenvalue

    def original_call(self, parameters, config):
        """
        Run the original call of the model with the given parameters and
        configuration by first rebuilding the model and then running it.
        Input:
        parameters (list): contains a list of parameters in this order:
            index 0: friction
            index 1: ... more can be included as needed,
                but other files need to be tweaked
        config (dict): contains resolution parameters for solver:
            time-step-relaxation (float): for global adaptive time steps,
                between 0 and 1
            min-depth (int): minimum depth for the simulation.
        """
        if not all(k in config for k in config_items):
            raise KeyError(f"config requires {config_items}")
        if config['time-step-relaxation'] <= 0 or\
            config['time-step-relaxation'] >= 1:
            raise ValueError(f"time-step-relaxation must be between 0 and 1 "
                             f"(exclusive).")
        elif not isinstance(config['min-depth'], int):
            raise ValueError(f"min-depth must be an integer.")
        os.system(
            f"cd sim_files && python3 coupled-tsunami-landslide.py "
            f"--friction {parameters[0][0]} "
            f"--time-step-relaxation {config['time-step-relaxation']} "
            f"--min-depth {config['min-depth']} "
            "> peano.txt"
        )
        os.system(f"cd sim_files && make -j$(nproc) > peano.txt")
        start = time.time()
        os.system(f"cd sim_files && ./CoupledLandslideTsunamis > peano.txt")
        end = time.time() - start
        return [[end]]

    def __call__(self, parameters, config):
        """
        Run the model with the given parameters and configuration.
        Input:
        parameters (list): contains a list of parameters in this order:
            index 0: friction
            index 1: ... more can be included as needed,
                but other files need to be tweaked
        config (dict): contains resolution parameters for solver:
            time-step-relaxation (float): for global adaptive time steps,
                between 0 and 1
            min-depth (int): minimum depth for the simulation.
        """
        if not all(k in config for k in config_items):
            raise KeyError(f"config requires {config_items}")
        if config['time-step-relaxation'] <= 0 or\
            config['time-step-relaxation'] >= 1:
            raise ValueError(f"time-step-relaxation must be between 0 and 1 "
                             f"(exclusive).")
        elif not isinstance(config['min-depth'], int):
            raise ValueError(f"min-depth must be an integer.")
        with open("sim_files/params.cpp", "w") as f:
            f.write("constexpr double invXi = " + str(parameters[0][0]) + ";")
        os.system(f"cd sim_files && make -j$(nproc) > peano.txt")
        os.system(f"cd sim_files && ./CoupledLandslideTsunamis > peano.txt")

    def supports_evaluate(self):
        return True
