import numpy as np


class Config(object):
    n_layer = 2
    batch_size = 5000 # original was 5000
    valid_size = 256 # original was 256
    step_boundaries = [2000, 4000]
    num_iterations = 2
    logging_frequency = 10 # original is 100
    verbose = True
    y_init_range = [0, 1]
    
class HJBMultiscaleConfig(Config):
    # Y_0 is about 4.5901.
    dim = 2
    total_time = 1.0
    num_time_interval = 80  #CHECK THIS DAVID
    lr_boundaries = [400]
    num_iterations = 3000
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [-0.05, 0]
    
    def setTotalTime(newTotalTime):
        total_time = newTotalTime
        
    def setNumTimeIntervals(newIntervals):
        num_time_interval = newIntervals


def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
