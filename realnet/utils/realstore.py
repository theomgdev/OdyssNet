"""
RealStore: Checkpoint Management Utilities for RealNet.

Provides standard save/load functionality plus "Weight Transplantation" 
for transferring knowledge between models of different sizes.
"""

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path, extra_data=None, trainer_state=None):
    """
    Saves a training checkpoint.
    
    Args:
        model: The RealNet model instance.
        optimizer: The optimizer instance.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        path (str): File path to save the checkpoint.
        extra_data (dict, optional): Any additional data to save.
        trainer_state (dict, optional): Runtime trainer state (scheduler/scaler/counters).
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if trainer_state is not None:
        checkpoint['trainer_state_dict'] = trainer_state
    
    if extra_data:
        checkpoint.update(extra_data)
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, optimizer, path, device='cpu', strict=True, lr=None, trainer=None):
    """
    Loads a training checkpoint.
    
    Args:
        model: The RealNet model instance (must match checkpoint architecture if strict=True).
        optimizer: The optimizer instance.
        path (str): File path to the checkpoint.
        device (str): Device to load tensors to.
        strict (bool): If True, raises error on architecture mismatch.
                       If False, ignores mismatched keys (standard PyTorch behavior).
        lr (float, optional): If provided, overwrites the learning rate in the optimizer 
                              after loading the state.
        trainer (optional): Trainer instance that implements load_state_dict.
    
    Returns:
        dict: The loaded checkpoint data (epoch, loss, etc.)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If strict=True and architecture doesn't match.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Overwrite LR if requested
            if lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    if 'initial_lr' in param_group:
                         param_group['initial_lr'] = lr
                print(f"⚡ Optimizer LR overwritten to: {lr}")
                
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e}. Optimizer will start fresh.")

    if trainer is not None and 'trainer_state_dict' in checkpoint:
        try:
            trainer.load_state_dict(checkpoint['trainer_state_dict'])
        except Exception as e:
            print(f"⚠️ Could not load trainer state: {e}. Runtime trainer state will start fresh.")
    
    return checkpoint


def transplant_weights(model, checkpoint_path, device='cpu', verbose=True, init_new='micro_quiet_8bit'):
    """
    🧬 Weight Transplantation: Transfer learned weights from a checkpoint to a model,
    even if the architectures (num_neurons) don't match.
    
    This is useful for:
    - Scaling up: Starting a larger model with knowledge from a smaller one.
    - Scaling down: Compressing a large model into a smaller one.
    - Warm starts: Any learned weights are better than random initialization.
    
    How it works:
    - For each parameter tensor, the overlapping region between source and target shapes is copied.
    - Core weight matrices (e.g., W and selected embed/proj/output_decoder weights) have their
      non-overlapping regions re-initialized using the `init_new` strategy (default: 'micro_quiet_8bit').
    - Biases and normalization parameters keep the initialization from the target model and are
      not re-initialized with `init_new`.
    
    Args:
        model: The target RealNet model instance (already initialized).
        checkpoint_path (str): Path to the source checkpoint.
        device (str): Device to load tensors to.
        verbose (bool): If True, prints transplant statistics.
        init_new (str): Weight init strategy for new (non-overlapping) regions of the core weights.
            Default 'micro_quiet_8bit' — stays silent while existing weights dominate,
            safe for fp16 AMP and 8-bit optimizers.
    
    Returns:
        dict: Statistics about the transplant operation.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get('model_state_dict', checkpoint)
    
    # Apply safe initialization strategy to core weights prior to region transplant.
    if init_new is not None and hasattr(model, '_apply_init'):
        with torch.no_grad():
            model._apply_init(model.W.data, init_new)
            if hasattr(model, 'embed') and model.embed is not None:
                model._apply_init(model.embed.weight.data, init_new)
            if hasattr(model, 'proj') and model.proj is not None:
                model._apply_init(model.proj.weight.data, init_new)
            if hasattr(model, 'output_decoder') and model.output_decoder is not None:
                model._apply_init(model.output_decoder.weight.data, init_new)
    
    target_state = model.state_dict()
    
    stats = {
        'total_params': 0,
        'transplanted_params': 0,
        'new_params': 0,
        'keys_matched': [],
        'keys_resized': [],
        'keys_missing': [],
    }
    
    for key in target_state:
        target_tensor = target_state[key]
        stats['total_params'] += target_tensor.numel()
        
        if key not in source_state:
            stats['keys_missing'].append(key)
            stats['new_params'] += target_tensor.numel()
            continue
            
        source_tensor = source_state[key]
        
        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.clone()
            stats['transplanted_params'] += target_tensor.numel()
            stats['keys_matched'].append(key)
            
        else:
            # Size mismatch - copy overlapping region
            stats['keys_resized'].append((key, source_tensor.shape, target_tensor.shape))
            
            if source_tensor.dim() == 2 and target_tensor.dim() == 2:
                min_rows = min(source_tensor.shape[0], target_tensor.shape[0])
                min_cols = min(source_tensor.shape[1], target_tensor.shape[1])
                target_state[key][:min_rows, :min_cols] = source_tensor[:min_rows, :min_cols]
                stats['transplanted_params'] += min_rows * min_cols
                stats['new_params'] += target_tensor.numel() - (min_rows * min_cols)
                
            elif source_tensor.dim() == 1 and target_tensor.dim() == 1:
                min_len = min(source_tensor.shape[0], target_tensor.shape[0])
                target_state[key][:min_len] = source_tensor[:min_len]
                stats['transplanted_params'] += min_len
                stats['new_params'] += target_tensor.numel() - min_len
                
            else:
                stats['keys_missing'].append(key)
                stats['new_params'] += target_tensor.numel()
    
    # Load the modified state
    model.load_state_dict(target_state)
    
    if verbose:
        print(f"🧬 Weight Transplantation Complete! (New regions: {init_new})")
        print(f"   Total Parameters: {stats['total_params']:,}")
        print(f"   Transplanted: {stats['transplanted_params']:,} ({100*stats['transplanted_params']/stats['total_params']:.1f}%)")
        print(f"   New (Initialized): {stats['new_params']:,} ({100*stats['new_params']/stats['total_params']:.1f}%)")
        
        if stats['keys_resized']:
            print("   Resized Tensors:")
            for key, src_shape, tgt_shape in stats['keys_resized']:
                print(f"      {key}: {list(src_shape)} → {list(tgt_shape)}")
    
    return stats


def get_checkpoint_info(path, device='cpu'):
    """
    Reads checkpoint metadata without loading into a model.
    
    Args:
        path (str): Path to the checkpoint file.
        device (str): Device to load tensors to.
    
    Returns:
        dict: Checkpoint information (epoch, loss, model size, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'loss': checkpoint.get('loss', 'Unknown'),
        'keys': list(checkpoint.get('model_state_dict', {}).keys()),
    }
    
    # Calculate model size
    if 'model_state_dict' in checkpoint:
        total_params = sum(
            t.numel() for t in checkpoint['model_state_dict'].values()
        )
        if 'W' in checkpoint['model_state_dict']:
            total_params -= checkpoint['model_state_dict']['W'].shape[0]
            
        info['total_params'] = total_params
        
        # Infer num_neurons from W shape
        if 'W' in checkpoint['model_state_dict']:
            info['num_neurons'] = checkpoint['model_state_dict']['W'].shape[0]
    
    return info
