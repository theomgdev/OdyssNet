import torch
import torch.nn as nn
import gc
import os
if os.environ.get('NO_BNB'):
    HAS_BNB = False
else:
    try:
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        # Check if user wants verbose logs (VERBOSE_BNB=1)
        if os.environ.get('VERBOSE_BNB'):
            import bitsandbytes as bnb
            HAS_BNB = True
        else:
            # Suppress bitsandbytes logs during import
            import sys
            _old_out, _old_err = sys.stdout, sys.stderr
            _null_out, _null_err = open(os.devnull, 'w'), open(os.devnull, 'w')
            try:
                sys.stdout, sys.stderr = _null_out, _null_err
                import bitsandbytes as bnb
                HAS_BNB = True
            finally:
                sys.stdout, sys.stderr = _old_out, _old_err
                _null_out.close()
                _null_err.close()
    except ImportError:
        HAS_BNB = False

class Neurogenesis:
    """
    Handles dynamic network growth (Neurogenesis) for RealNet models.
    """

    @staticmethod
    def expand(model, optimizer, amount=1, verbose=True):
        """
        Dynamically adds neurons to the network while preserving state and memory.
        
        Args:
            model (RealNet): The model to expand.
            optimizer (torch.optim.Optimizer): The current optimizer.
            amount (int): Number of neurons to add.
            verbose (bool): Whether to print status.
            
        Returns:
            torch.optim.Optimizer: The new optimizer instance.
        """
        if verbose:
            print(f"\n🌱 Neurogenesis: Growing Network {model.num_neurons} -> {model.num_neurons + amount} Neurons")
        
        old_n = model.num_neurons
        new_n = old_n + amount
        device = model.device
        
        # Preserve old parameters
        old_W_param = model.W
        old_B_param = model.B
        
        # Norm Preservation (StepNorm)
        old_norm_w_param = model.norm.weight
        old_norm_b_param = model.norm.bias
        
        old_input_scale = model.input_scale
        old_output_scale = model.output_scale
        
        # Explicitly preserve ID lists (vital for offsets)
        old_input_ids = list(model.input_ids)
        old_output_ids = list(model.output_ids)
        
        old_opt = optimizer
        
        # Expand W
        new_W = torch.zeros(new_n, new_n, device=device)
        new_W[:old_n, :old_n] = model.W.data
        
        # micro_quiet_8bit init for new connections
        noise_std = 1e-3
        new_W[:old_n, old_n:] = torch.randn(old_n, amount, device=device) * noise_std
        new_W[old_n:, :old_n] = torch.randn(amount, old_n, device=device) * noise_std
        new_W[old_n:, old_n:] = torch.randn(amount, amount, device=device) * noise_std
        
        # Expand B
        new_B = torch.zeros(new_n, device=device)
        new_B[:old_n] = model.B.data

        # Expand LayerNorm
        new_norm = nn.LayerNorm(new_n).to(device)
        
        with torch.no_grad():
            new_norm.weight[:old_n] = model.norm.weight.data
            new_norm.bias[:old_n]   = model.norm.bias.data

        # Expand State
        if hasattr(model, 'state'):
            new_state = torch.zeros(model.state.shape[0], new_n, device=device)
            new_state[:, :old_n] = model.state
            model.state = new_state
            
        # Apply changes
        model.num_neurons = new_n
        model.W = nn.Parameter(new_W)
        model.B = nn.Parameter(new_B)
        
        # StepNorm
        model.norm = new_norm
            
        # Scaling params
        model.input_scale = nn.Parameter(old_input_scale.data)
        model.output_scale = nn.Parameter(old_output_scale.data)
        
        # Re-bind IDs and update buffers
        model.input_ids = old_input_ids
        model.output_ids = old_output_ids
        model.register_buffer('input_pos', torch.tensor(old_input_ids, dtype=torch.long, device=device))
        model.register_buffer('output_pos', torch.tensor(old_output_ids, dtype=torch.long, device=device))
        
        # Optimizer Migration
        group = old_opt.param_groups[0]
        optimizer_cls = type(old_opt)
        
        # Helper to safely get arg
        def get_arg(name, default):
            return group.get(name, default)

        # Re-instantiate optimizer
        try:
             new_opt = optimizer_cls(
                model.parameters(), 
                lr=get_arg('lr', 0.001), 
                weight_decay=get_arg('weight_decay', 0), 
                betas=get_arg('betas', (0.9, 0.999)), 
                eps=get_arg('eps', 1e-8)
            )
        except Exception as e:
            print(f"⚠️ Optimizer re-init failed: {e}. Falling back to standard AdamW.")
            new_opt = torch.optim.AdamW(model.parameters(), lr=group['lr'])

        # Migrate internal optimizer state
        def transfer_state(old_p, new_p, is_matrix=False):
            if old_p in old_opt.state:
                old_s = old_opt.state[old_p]
                new_s = {}
                
                # Copy scalar attributes (step, etc.)
                for k, v in old_s.items():
                    if not torch.is_tensor(v):
                        new_s[k] = v
                
                # Iterate over ALL keys in state (including qmap1, absmax1, etc.)
                for key, val in old_s.items():
                    if not torch.is_tensor(val):
                         # Already copied scalars above
                         continue
                         
                    try:
                        tensor = val
                        if tensor.shape == old_p.shape:
                            # Needs resizing
                            new_tensor = torch.zeros(new_p.shape, dtype=tensor.dtype, device=device)
                            
                            if is_matrix:
                                min_rows = min(tensor.shape[0], new_p.shape[0])
                                min_cols = min(tensor.shape[1], new_p.shape[1])
                                new_tensor[:min_rows, :min_cols] = tensor[:min_rows, :min_cols]
                            else:
                                min_len = min(tensor.shape[0], new_p.shape[0])
                                new_tensor[:min_len] = tensor[:min_len]
                                
                            new_s[key] = new_tensor
                            
                        else:
                            # Metadata tensor - direct copy
                            new_s[key] = tensor.clone()
                            
                    except Exception as e:
                        print(f"      Running into issue with key '{key}': {e}. Skipping.")
                        pass
                
                # Assign gathered state to new optimizer
                if new_s:
                    new_opt.state[new_p] = new_s
                
        # Transfer state for each parameter
        is_bnb = HAS_BNB and isinstance(new_opt, (bnb.optim.Adam8bit, bnb.optim.AdamW8bit))
        
        if is_bnb:
             print("   👉 BNB Optimizer Detected. Skipping State Transfer (Cold Restart) to avoid quantization explosion.")
        else:
            try:
                transfer_state(old_W_param, model.W, is_matrix=True)
                transfer_state(old_B_param, model.B, is_matrix=False)
                
                # Transfer StepNorm State
                transfer_state(old_norm_w_param, model.norm.weight, is_matrix=False)
                transfer_state(old_norm_b_param, model.norm.bias, is_matrix=False)
                
                transfer_state(old_input_scale, model.input_scale, is_matrix=False)
                transfer_state(old_output_scale, model.output_scale, is_matrix=False)
                
                # Transfer Vocab Layers State (Shapes are constant during Neurogenesis)
                if hasattr(model, 'embed') and model.embed is not None:
                     transfer_state(model.embed.weight, model.embed.weight, is_matrix=True)
                
                if hasattr(model, 'proj') and model.proj is not None:
                     transfer_state(model.proj.weight, model.proj.weight, is_matrix=True)
                     
                if hasattr(model, 'output_decoder') and model.output_decoder is not None:
                     transfer_state(model.output_decoder.weight, model.output_decoder.weight, is_matrix=True)
                
                print("   ✅ Optimizer State Transferred (Momentum Preserved)")
            except Exception as e:
                print(f"   ⚠️ Optimizer State Transfer Failed ({e}). Performing Cold Restart.")

        # Cleanup
        del old_W_param
        del old_B_param
        del old_norm_w_param
        del old_norm_b_param
        del old_opt
        del old_input_scale
        del old_output_scale
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if verbose:
            print(f"✅ Neurogenesis Complete. Optimizer Migrated. New Size: {new_n}. VRAM Cleared.")
            
        return new_opt
