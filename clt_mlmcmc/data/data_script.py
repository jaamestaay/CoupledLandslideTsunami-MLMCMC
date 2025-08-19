import argparse
import numpy as np

from clt_mlmcmc.data.process_data import forward_model
from clt_mlmcmc.classes.umbridge_server import CoupledTsunamiLandslide

parser = argparse.ArgumentParser(description='Model output test.')
args = parser.parse_args()
model = CoupledTsunamiLandslide()

original_config = {
        'level': 0,
        'samples': 1000,
        'time-step-relaxation': 0.6,
        'min-depth': 4,
        'noise_std': 1.2,
        'proposal_std': 0.001
    }

# Original Data Generation
data = forward_model(1/200.0, original_config, model, 'h1u')
np.savetxt('observed_data.csv', data, delimiter=',')
data += np.random.normal(0, scale=0.01, size=data.size)
np.savetxt('observed_data_noise.csv', data, delimiter=',')