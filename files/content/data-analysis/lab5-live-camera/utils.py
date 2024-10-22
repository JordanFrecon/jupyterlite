def count_mlp_parameters(model):
    """
    Compute the number of parameters in a MLPClassifier model.

    Parameters:
    - model: MLPClassifier object.

    Returns:
    - num_params: Total number of parameters in the model.
    """
    num_params = 0
    for layer_idx in range(model.n_layers_ - 1):
        # Counting parameters for weights
        num_params += (model.coefs_[layer_idx].shape[0] + 1) * model.coefs_[layer_idx].shape[1]
        # Counting parameters for biases
        num_params += model.coefs_[layer_idx].shape[1]

    return num_params

