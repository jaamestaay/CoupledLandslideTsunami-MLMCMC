import argparse
import time
import numpy as np

from process_data import forward_model
from umbridge_server import CoupledTsunamiLandslide

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

# # Original Data Generation
# data = forward_model(1/200.0, original_config, model, 'h1u')
# np.savetxt('observed_data.csv', data, delimiter=',')
# data += np.random.normal(0, scale=0.01, size=data.size)
# np.savetxt('observed_data_noise.csv', data, delimiter=',')

theta = 0.005
# start = time.time()
# output = forward_model(theta, original_config, model, 'h1u', False)
# end = time.time()
# print(end-start)

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

y_obs = np.loadtxt('observed_data_noise.csv', delimiter=',')
for i in range(3):
    forward_model(theta, multilevel_config[i], model, 'h1u', True)
    forward_model(theta, multilevel_config[i], model, 'h1u', False)