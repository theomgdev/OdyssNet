import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    Sets a fixed seed for reproducible results across all random sources.
    
    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            if num_assigned > 0:
                target_ids = model_input_ids[:num_assigned]
                x_input[:, :, target_ids] = input_features[:, :, :num_assigned]
            
            return x_input, batch_size

        # Handle case where input_features might be 1D (Batch,) -> (Batch, 1)
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(1)
            
        num_features = input_features.shape[1]
        num_assigned = min(num_features, len(model_input_ids))
        
        # Assign features to neurons
        if num_assigned > 0:
            target_ids = model_input_ids[:num_assigned]
            x_input[:, target_ids] = input_features[:, :num_assigned]
            
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
        # Promote float64 to float32 by default; preserve integer dtypes as-is for index/token inputs.
        if dtype == torch.float64:
            dtype = torch.float32

    return torch.as_tensor(data, dtype=dtype, device=device)
