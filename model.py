#number of passes through the entire training dataset
#lr-leasrning rate
def model(X, Y, layer_dim, epochs, lr):
    params = init_params(layer_dim)
    cost_history = []
    
    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        
        params = update_parameters(params, grads, lr)
        
        
    return params, cost_history