import torch

def prepare_input(input_features, model_input_ids, num_neurons, device):
    """
    Maps input features (Batch, Input_Size) to the full neuron tensor (Batch, N).
    
    Args:
        input_features (Tensor or np.array): Raw input data.
        model_input_ids (list): List of neuron indices that accept input.
        num_neurons (int): Total number of neurons in the model.
        device (str or torch.device): Device to put the tensor on.
        
    Returns:
        Tensor: Prepared input tensor of shape (Batch, Num_Neurons).
    """
    # Convert to Tensor
    if not isinstance(input_features, torch.Tensor):
        input_features = torch.tensor(input_features, dtype=torch.float32, device=device)
    else:
        input_features = input_features.to(device)
        
    batch_size = input_features.shape[0]
    
    # Initialize Full Neuron Tensor
    x_input = torch.zeros(batch_size, num_neurons, device=device)

    # Map features to input neurons
    if len(model_input_ids) > 0:
        # Check for Sequential Input (Batch, Steps, Features)
        if input_features.dim() == 3:
            # (Batch, Steps, Features) -> (Batch, Steps, Num_Neurons)
            batch_size, steps, num_features = input_features.shape
            x_input = torch.zeros(batch_size, steps, num_neurons, device=device)
            
            num_assigned = min(num_features, len(model_input_ids))
            for k in range(num_assigned):
                x_input[:, :, model_input_ids[k]] = input_features[:, :, k]
            
            return x_input, batch_size

        # Handle case where input_features might be 1D (Batch,) -> (Batch, 1)
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(1)
            
        num_features = input_features.shape[1]
        num_assigned = min(num_features, len(model_input_ids))
        
        # Assign features to neurons
        for k in range(num_assigned):
            x_input[:, model_input_ids[k]] = input_features[:, k]
            
    return x_input, batch_size

def to_tensor(data, device, dtype=None):
    """
    Safely converts data to a PyTorch tensor on the target device.
    """
    if isinstance(data, torch.Tensor):
        if dtype is not None:
            return data.to(device=device, dtype=dtype)
        return data.to(device)

    if dtype is None:
        inferred = torch.as_tensor(data)
        dtype = inferred.dtype
        # Keep default project behavior for float inputs while preserving integer targets/tokens.
        if dtype == torch.float64:
            dtype = torch.float32

    return torch.as_tensor(data, dtype=dtype, device=device)
