import numpy as np
def init_params(layer_dim):
    np.random.seed(3)
    params = {}
    L = len(layer_dim)
    
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dim[l], layer_dim[l-1])*0.01
        params['b'+str(l)] = np.zeros((layer_dim[l], 1))
        
    return params