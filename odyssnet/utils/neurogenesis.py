import torch
import torch.nn as nn

class Neurogenesis:
    """
    Handles dynamic network growth (Neurogenesis) for OdyssNet models.
    """

    @staticmethod
    def expand(model, optimizer, amount=1, verbose=True):
        """
        Dynamically adds neurons to the network while preserving state and memory.
        
        Args:
            model (OdyssNet): The model to expand.
            optimizer (torch.optim.Optimizer): The current optimizer.
            amount (int): Number of neurons to add.
            verbose (bool): Whether to print status.

        Returns:
            torch.optim.Optimizer: The new optimizer instance.
        """
        if verbose:
            print(f"\nNeurogenesis: Growing Network {model.num_neurons} -> {model.num_neurons + amount} Neurons")
        
        old_n = model.num_neurons
        new_n = old_n + amount
        device = model.device
        
        # Preserve old parameters
        old_W_param = model.W
        old_B_param = model.B
        old_memory_param = model.memory_feedback
        
        # Norm Preservation (StepNorm)
        old_norm_w_param = model.norm.weight
        old_norm_b_param = getattr(model.norm, 'bias', None)
        
        old_input_scale = model.input_scale
        old_output_scale = model.output_scale

        # Optional gate parameters (present when gate branch is enabled)
        old_input_gate_param = getattr(model, 'input_gate', None)
        old_output_gate_param = getattr(model, 'output_gate', None)
        old_core_gate_param = getattr(model, 'core_gate', None)
        old_memory_gate_param = getattr(model, 'memory_gate', None)
        gate_init_strategy = getattr(model, 'gate_weight_init', 'zero')

        # Preserve Hebbian parameters for optimizer state transfer.
        # For "neuron" and "synapse" types the params are vectors/matrices and are
        # zero-padded to the new size; "global" scalars are shape-invariant.
        hebb_type = getattr(model, 'hebb_type', None)
        hebb_res = getattr(model, 'hebb_res', None)
        def _get_hebb(name): return getattr(model, name, None)

        old_t_hebb_factor = _get_hebb('t_hebb_factor')
        old_t_hebb_decay  = _get_hebb('t_hebb_decay')
        old_s_hebb_factor = _get_hebb('s_hebb_factor')
        old_s_hebb_decay  = _get_hebb('s_hebb_decay')
        
        old_t_hebb_state_W   = _get_hebb('t_hebb_state_W').clone() if _get_hebb('t_hebb_state_W') is not None else None
        old_t_hebb_state_mem = _get_hebb('t_hebb_state_mem').clone() if _get_hebb('t_hebb_state_mem') is not None else None
        old_s_hebb_state_W   = _get_hebb('s_hebb_state_W').clone() if _get_hebb('s_hebb_state_W') is not None else None
        old_s_hebb_state_mem = _get_hebb('s_hebb_state_mem').clone() if _get_hebb('s_hebb_state_mem') is not None else None

        # Preserve vocab layer parameters (for optimizer state transfer)
        old_embed_param = model.embed.weight if model.embed is not None else None
        old_proj_param = model.proj.weight if model.proj is not None else None
        old_decoder_param = model.output_decoder.weight if model.output_decoder is not None else None

        # Explicitly preserve ID lists (vital for offsets)
        old_input_ids = list(model.input_ids)
        old_output_ids = list(model.output_ids)
        
        old_opt = optimizer
        
        # Expand W
        new_W = torch.zeros(new_n, new_n, device=device)
        new_W[:old_n, :old_n] = model.W.data
        
        # micro_quiet_warm init for new connections
        noise_std = 1e-3
        new_W[:old_n, old_n:] = torch.randn(old_n, amount, device=device) * noise_std
        new_W[old_n:, :old_n] = torch.randn(amount, old_n, device=device) * noise_std
        new_W[old_n:, old_n:] = torch.randn(amount, amount, device=device) * noise_std
        new_W.fill_diagonal_(0.0)
        
        # Expand B
        new_B = torch.zeros(new_n, device=device)
        new_B[:old_n] = model.B.data
        
        # Expand memory_feedback
        new_memory = torch.zeros(new_n, device=device)
        new_memory[:old_n] = model.memory_feedback.data
        new_memory[old_n:] = torch.randn(amount, device=device) * noise_std

        # Preserve original normalization family to avoid training-dynamics drift.
        if isinstance(model.norm, nn.RMSNorm):
            new_norm = nn.RMSNorm(new_n, eps=model.norm.eps).to(device)
        elif isinstance(model.norm, nn.LayerNorm):
            new_norm = nn.LayerNorm(new_n, eps=model.norm.eps).to(device)
        else:
            new_norm = nn.LayerNorm(new_n).to(device)

        with torch.no_grad():
            new_norm.weight[:old_n] = model.norm.weight.data
            old_norm_bias = getattr(model.norm, 'bias', None)
            new_norm_bias = getattr(new_norm, 'bias', None)
            if isinstance(old_norm_bias, torch.Tensor) and isinstance(new_norm_bias, torch.Tensor):
                new_norm_bias[:old_n] = old_norm_bias.data

        # Expand State
        if hasattr(model, 'state'):
            new_state = torch.zeros(model.state.shape[0], new_n, device=device)
            new_state[:, :old_n] = model.state
            model.state = new_state
            
        # Apply changes
        model.num_neurons = new_n
        model.W = nn.Parameter(new_W)
        model.B = nn.Parameter(new_B)
        model.memory_feedback = nn.Parameter(new_memory)

        # Preserve W diagonal constraints after replacing parameter
        with torch.no_grad():
            model.W.fill_diagonal_(0.0)

        def _zero_diagonal_grad(grad):
            return grad.clone().fill_diagonal_(0.0)
        model.W.register_hook(_zero_diagonal_grad)
        
        # StepNorm
        model.norm = new_norm
            
        # Scaling params
        model.input_scale = nn.Parameter(old_input_scale.data)
        model.output_scale = nn.Parameter(old_output_scale.data)

        # Recreate gate params so optimizer can safely rebind state.
        if isinstance(old_input_gate_param, nn.Parameter):
            model.input_gate = nn.Parameter(old_input_gate_param.data.clone())
        else:
            model.input_gate = None

        if isinstance(old_output_gate_param, nn.Parameter):
            model.output_gate = nn.Parameter(old_output_gate_param.data.clone())
        else:
            model.output_gate = None

        if isinstance(old_core_gate_param, nn.Parameter):
            new_core_gate = torch.empty(new_n, device=device)
            new_core_gate[:old_n] = old_core_gate_param.data
            if amount > 0:
                tail = torch.empty(amount, device=device)
                if hasattr(model, '_apply_init'):
                    model._apply_init(tail, gate_init_strategy)
                else:
                    nn.init.zeros_(tail)
                new_core_gate[old_n:] = tail
            model.core_gate = nn.Parameter(new_core_gate)
        else:
            model.core_gate = None

        if isinstance(old_memory_gate_param, nn.Parameter):
            new_memory_gate = torch.empty(new_n, device=device)
            new_memory_gate[:old_n] = old_memory_gate_param.data
            if amount > 0:
                tail = torch.empty(amount, device=device)
                if hasattr(model, '_apply_init'):
                    model._apply_init(tail, gate_init_strategy)
                else:
                    nn.init.zeros_(tail)
                new_memory_gate[old_n:] = tail
            model.memory_gate = nn.Parameter(new_memory_gate)
        else:
            model.memory_gate = None

        if hasattr(model, '_cached_scaled_input'):
            delattr(model, '_cached_scaled_input')

        # Resize Hebbian buffers to match the new neuron count.
        # The existing (old_n × old_n) correlation state is copied into the top-left
        # corner of the new (new_n × new_n) buffer; the new rows/cols start at zero.
        if hebb_type is not None:
            def _resize_buffer(old_buffer, is_matrix=False):
                if old_buffer is None: return None
                if is_matrix:
                    new_b = torch.zeros(new_n, new_n, device=device)
                    new_b[:old_n, :old_n] = old_buffer
                else:
                    new_b = torch.zeros(new_n, device=device)
                    new_b[:old_n] = old_buffer
                return new_b

            if old_t_hebb_state_W is not None:
                model.register_buffer('t_hebb_state_W', _resize_buffer(old_t_hebb_state_W, True))
                model.register_buffer('t_hebb_state_mem', _resize_buffer(old_t_hebb_state_mem, False))
            if old_s_hebb_state_W is not None:
                model.register_buffer('s_hebb_state_W', _resize_buffer(old_s_hebb_state_W, True))
                model.register_buffer('s_hebb_state_mem', _resize_buffer(old_s_hebb_state_mem, False))

            def _resize_param(old_param, default_val):
                if old_param is None: return None
                if hebb_res == "neuron":
                    new_p = torch.full((new_n,), default_val, device=device)
                    new_p[:old_n] = old_param.data
                elif hebb_res == "synapse":
                    new_p = torch.full((new_n, new_n), default_val, device=device)
                    new_p[:old_n, :old_n] = old_param.data
                else: # global
                    return nn.Parameter(old_param.data.clone())
                return nn.Parameter(new_p)

            if old_t_hebb_factor is not None:
                model.t_hebb_factor = _resize_param(old_t_hebb_factor, -3.0)
                model.t_hebb_decay  = _resize_param(old_t_hebb_decay, 2.2)
            if old_s_hebb_factor is not None:
                model.s_hebb_factor = _resize_param(old_s_hebb_factor, -3.0)
                model.s_hebb_decay  = _resize_param(old_s_hebb_decay, 2.2)

        # Re-bind IDs and update buffers
        model.input_ids = old_input_ids
        model.output_ids = old_output_ids
        model.register_buffer('input_pos', torch.tensor(old_input_ids, dtype=torch.long, device=device))
        model.register_buffer('output_pos', torch.tensor(old_output_ids, dtype=torch.long, device=device))
        
        # Optimizer Migration
        group = old_opt.param_groups[0]
        optimizer_cls = type(old_opt)

        try:
            new_opt = optimizer_cls(
                model.parameters(),
                lr=group.get('lr', 0.001),
                weight_decay=group.get('weight_decay', 0),
                betas=group.get('betas', (0.9, 0.999)),
                eps=group.get('eps', 1e-8),
            )
        except Exception as e:
            print(f"WARNING: Optimizer re-init failed: {e}. Falling back to standard AdamW.")
            new_opt = torch.optim.AdamW(model.parameters(), lr=group.get('lr', 0.001))

        # Migrate internal optimizer state
        def transfer_state(old_p, new_p, is_matrix=False):
            if old_p not in old_opt.state:
                return

            old_s = old_opt.state[old_p]
            new_s = {}

            for key, val in old_s.items():
                if not torch.is_tensor(val):
                    # Scalar attributes (step counter, etc.) — copy as-is.
                    new_s[key] = val
                    continue

                try:
                    if val.shape != old_p.shape:
                        # Tensor needs resizing to match the expanded parameter.
                        new_tensor = torch.zeros(new_p.shape, dtype=val.dtype, device=device)
                        if is_matrix:
                            min_rows = min(val.shape[0], new_p.shape[0])
                            min_cols = min(val.shape[1], new_p.shape[1])
                            new_tensor[:min_rows, :min_cols] = val[:min_rows, :min_cols]
                        else:
                            min_len = min(val.shape[0], new_p.shape[0])
                            new_tensor[:min_len] = val[:min_len]
                        new_s[key] = new_tensor
                    else:
                        # Shape-invariant metadata tensor — direct copy.
                        new_s[key] = val.clone()
                except Exception as e:
                    print(f"      Skipping optimizer state key '{key}': {e}")

            if new_s:
                new_opt.state[new_p] = new_s
                
        # Transfer state for each parameter
        try:
            transfer_state(old_W_param, model.W, is_matrix=True)
            transfer_state(old_B_param, model.B, is_matrix=False)
            transfer_state(old_memory_param, model.memory_feedback, is_matrix=False)

            # Transfer StepNorm state
            transfer_state(old_norm_w_param, model.norm.weight, is_matrix=False)
            if old_norm_b_param is not None and getattr(model.norm, 'bias', None) is not None:
                transfer_state(old_norm_b_param, model.norm.bias, is_matrix=False)

            transfer_state(old_input_scale, model.input_scale, is_matrix=False)
            transfer_state(old_output_scale, model.output_scale, is_matrix=False)

            if isinstance(old_input_gate_param, nn.Parameter) and isinstance(getattr(model, 'input_gate', None), nn.Parameter):
                transfer_state(old_input_gate_param, model.input_gate, is_matrix=False)
            if isinstance(old_output_gate_param, nn.Parameter) and isinstance(getattr(model, 'output_gate', None), nn.Parameter):
                transfer_state(old_output_gate_param, model.output_gate, is_matrix=False)
            if isinstance(old_core_gate_param, nn.Parameter) and isinstance(getattr(model, 'core_gate', None), nn.Parameter):
                transfer_state(old_core_gate_param, model.core_gate, is_matrix=False)
            if isinstance(old_memory_gate_param, nn.Parameter) and isinstance(getattr(model, 'memory_gate', None), nn.Parameter):
                transfer_state(old_memory_gate_param, model.memory_gate, is_matrix=False)

            # Transfer Hebbian parameter optimizer state.
            # "global" scalars are shape-invariant; "neuron"/"synapse" params were
            # padded above so the old state is copied into the overlapping region.
            if hebb_type is not None:
                def _transfer_hebb(old_p, new_p_name):
                    new_p = getattr(model, new_p_name, None)
                    if isinstance(old_p, nn.Parameter) and isinstance(new_p, nn.Parameter):
                        transfer_state(old_p, new_p, is_matrix=(hebb_res == "synapse"))
                
                _transfer_hebb(old_t_hebb_factor, 't_hebb_factor')
                _transfer_hebb(old_t_hebb_decay, 't_hebb_decay')
                _transfer_hebb(old_s_hebb_factor, 's_hebb_factor')
                _transfer_hebb(old_s_hebb_decay, 's_hebb_decay')

            # Transfer vocab layer state (shapes constant during neurogenesis)
            if old_embed_param is not None and hasattr(model, 'embed') and model.embed is not None:
                transfer_state(old_embed_param, model.embed.weight, is_matrix=True)
            if old_proj_param is not None and hasattr(model, 'proj') and model.proj is not None:
                transfer_state(old_proj_param, model.proj.weight, is_matrix=True)
            if old_decoder_param is not None and hasattr(model, 'output_decoder') and model.output_decoder is not None:
                transfer_state(old_decoder_param, model.output_decoder.weight, is_matrix=True)

            print("   OK: Optimizer state transferred (momentum preserved)")
        except Exception as e:
            print(f"   WARNING: Optimizer state transfer failed ({e}). Performing cold restart.")

        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(f"Neurogenesis complete. Optimizer migrated. New size: {new_n}. VRAM cleared.")

        return new_opt
