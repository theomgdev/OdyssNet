import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import cast

class OdyssNet(nn.Module):
    def __init__(self, num_neurons, input_ids, output_ids, pulse_mode=True, dropout_rate=0.0, device='cpu', weight_init=None, activation=None, gradient_checkpointing=False, vocab_size=None, vocab_mode='hybrid', tie_embeddings=False, gate=None, hebb_type=None, debug=False):
        super(OdyssNet, self).__init__()
        
        # Auto-size to unique input+output IDs
        if num_neurons == -1:
            unique_ids = set(input_ids) | set(output_ids)
            if len(unique_ids) > 0:
                max_id = max(unique_ids)
                num_neurons = max_id + 1

                difference = num_neurons - len(unique_ids)
                if difference > 0:
                    print(f"OdyssNet Auto-Sizing: Sparse IDs detected. Created {num_neurons} neurons (covering Max ID {max_id}). Unconnected neurons: {difference}")
            else:
                num_neurons = 0
        
        self.num_neurons = num_neurons
        self.debug = debug
        if debug:
            torch.autograd.set_detect_anomaly(True)

        self.input_ids = input_ids
        self.output_ids = output_ids
        
        # Vocab / Projection Mode
        self.vocab_size = vocab_size
        self.embed = None
        self.proj = None
        self.output_decoder = None

        if vocab_size is not None:
            # Parse Asymmetric Vocab Size
            if isinstance(vocab_size, (list, tuple)):
                v_in, v_out = vocab_size
            else:
                v_in = vocab_size
                v_out = vocab_size

            # Output Decoder (Neurons -> Vocab)
            # Enabled if v_out > 0
            if v_out > 0:
                self.output_decoder = nn.Linear(len(output_ids), v_out, bias=False).to(device)
            
            # Input Projection (Vocab -> Neurons)
            # Enabled if v_in > 0
            if v_in > 0:
                target_dim = len(input_ids)
                
                if vocab_mode in ['hybrid', 'discrete']:
                    self.embed = nn.Embedding(v_in, target_dim).to(device)
                    
                if vocab_mode in ['hybrid', 'continuous']:
                    self.proj = nn.Linear(v_in, target_dim, bias=False).to(device)

            # Weight Tying (Embeddings -> Decoder)
            if tie_embeddings and (v_in == v_out) and (len(input_ids) == len(output_ids)):
                if self.embed is not None and self.output_decoder is not None:
                    self.output_decoder.weight = self.embed.weight
                elif self.proj is not None and self.embed is None:
                    print("WARNING: Weight tying is not supported for 'continuous' (Linear) vocab_mode due to transposed dimensions.")

        # Buffers for fast indexing
        self.register_buffer('input_pos', torch.tensor(input_ids, dtype=torch.long, device=device))
        self.register_buffer('output_pos', torch.tensor(output_ids, dtype=torch.long, device=device))
        
        # Scaling Parameters (Input/Output)
        self.input_scale = nn.Parameter(torch.full((len(input_ids),), 1.0, device=device))
        self.output_scale = nn.Parameter(torch.full((len(output_ids),), 1.0, device=device))
        
        self.pulse_mode = pulse_mode
        self.gradient_checkpointing = gradient_checkpointing
        self._cached_scaled_input = None
        
        # Parse configurable component settings
        weight_init = self._normalize_weight_init(weight_init)
        activation = self._normalize_activation(activation)
        gate = self._normalize_gate(gate)

        self.enc_dec_weight_init, self.core_weight_init, self.mem_weight_init, self.gate_weight_init = weight_init
        self.weight_init_strategy = self.core_weight_init

        self.enc_dec_act = self._build_activation(activation[0])
        self.act = self._build_activation(activation[1])
        self.mem_act = self._build_activation(activation[2])

        self.enc_dec_gate_act = self._build_gate_activation(gate[0])
        self.core_gate_act = self._build_gate_activation(gate[1])
        self.mem_gate_act = self._build_gate_activation(gate[2])
        
        # Weight Matrix (N x N)
        self.W = nn.Parameter(torch.empty(num_neurons, num_neurons, device=device))

        # Learnable gate parameters (enabled only for non-'none' gate entries)
        self.input_gate = self._create_gate_parameter(len(input_ids), self.enc_dec_gate_act, device)
        self.output_gate = self._create_gate_parameter(len(output_ids), self.enc_dec_gate_act, device)
        self.core_gate = self._create_gate_parameter(num_neurons, self.core_gate_act, device)
        self.memory_gate = self._create_gate_parameter(num_neurons, self.mem_gate_act, device)

        self._init_weights()
        
        # Memory Feedback (Neuron self-connections)
        self.memory_feedback = nn.Parameter(torch.empty(num_neurons, device=device))
        with torch.no_grad():
            self._apply_init(self.memory_feedback, self.mem_weight_init)
            self.W.fill_diagonal_(0.0)

        def _zero_diagonal_grad(grad):
            return grad.clone().fill_diagonal_(0.0)
        self.W.register_hook(_zero_diagonal_grad)

        # Hebbian Learning (optional)
        # hebb_type controls the structural resolution of plasticity:
        #   None       — plasticity disabled (default).
        #   "global"   — single scalar factor and decay.
        #   "neuron"   — per-neuron (N,) vector factor and decay (+2N params).
        #   "synapse"  — per-synapse (N,N) matrix factor and decay (+2N² params).
        # hebb_factor and hebb_decay are raw logits; sigmoid maps them to (0, 1).
        # sigmoid(-3.0) ≈ 0.047 — small initial Hebbian influence on W and memory_feedback.
        # sigmoid( 2.2) ≈ 0.900 — high initial retention of accumulated correlations.

        if hebb_type not in (None, "global", "neuron", "synapse"):
            raise ValueError(
                f"hebb_type must be None, 'global', 'neuron', or 'synapse', got {hebb_type!r}"
            )

        self.hebb_type = hebb_type

        if hebb_type is not None:
            if hebb_type == "global":
                self.hebb_factor = nn.Parameter(torch.full((), -3.0, device=device))
                self.hebb_decay  = nn.Parameter(torch.full((),  2.2, device=device))
            elif hebb_type == "neuron":
                self.hebb_factor = nn.Parameter(torch.full((num_neurons,), -3.0, device=device))
                self.hebb_decay  = nn.Parameter(torch.full((num_neurons,),  2.2, device=device))
            elif hebb_type == "synapse":
                self.hebb_factor = nn.Parameter(torch.full((num_neurons, num_neurons), -3.0, device=device))
                self.hebb_decay  = nn.Parameter(torch.full((num_neurons, num_neurons),  2.2, device=device))
            # Accumulated cross-neuron correlations (diagonal kept zero, mirrors W constraint).
            self.register_buffer('hebb_state_W',   torch.zeros(num_neurons, num_neurons, device=device))
            # Accumulated self-correlations for memory_feedback (diagonal of the outer product).
            self.register_buffer('hebb_state_mem', torch.zeros(num_neurons, device=device))
        else:
            self.hebb_factor = None
            self.hebb_decay  = None

        # Bias Vector
        self.B = nn.Parameter(torch.zeros(num_neurons, device=device))

        # Norm
        self.norm = nn.RMSNorm(num_neurons).to(device)
        
        self.drop = nn.Dropout(p=dropout_rate)

        # Internal State (hidden state h_t)
        self.state = torch.zeros(1, num_neurons, device=device)
        
    def _dbg(self, tensor, label):
        # Guard must be at the call site (`if self.debug: self._dbg(...)`) not only here.
        # Python evaluates f-string arguments before entering this function, so an internal
        # check would still build every label string on every step when debug=False.
        if self.debug and not torch.isfinite(tensor).all():
            n_nan = tensor.isnan().sum().item()
            n_inf = tensor.isinf().sum().item()
            raise RuntimeError(
                f"NaN/Inf — {label} | "
                f"nan={n_nan} inf={n_inf} shape={list(tensor.shape)} dtype={tensor.dtype}"
            )

    def _build_activation(self, name):
        if name is None:
            return nn.Identity()

        key = name.lower() if isinstance(name, str) else name
        if key == 'none' or key == 'identity':
            return nn.Identity()
        elif key == 'tanh':
            return nn.Tanh()
        elif key == 'relu':
            return nn.ReLU()
        elif key == 'leaky_relu':
            return nn.LeakyReLU()
        elif key == 'sigmoid':
            return nn.Sigmoid()
        elif key == 'gelu':
            return nn.GELU()
        elif key == 'gelu_tanh':
            return nn.GELU(approximate='tanh')
        elif key == 'silu':
             return nn.SiLU()
        else:
             raise ValueError(f"Unknown activation function: {name}")

    def _normalize_component_list(self, value, defaults, name):
        if value is None:
            return defaults.copy()

        if isinstance(value, (list, tuple)):
            values = list(value)
            if len(values) == 0:
                raise ValueError(f"{name} list cannot be empty")
            if len(values) > len(defaults):
                raise ValueError(f"{name} list supports at most {len(defaults)} items, got {len(values)}")

            normalized = defaults.copy()
            normalized[:len(values)] = values
            return normalized

        raise TypeError(f"{name} must be None, str, list, or tuple")

    def _normalize_weight_init(self, weight_init):
        defaults = ['quiet', 'resonant', 'quiet', 'zero']
        if weight_init is None:
            return defaults.copy()

        if isinstance(weight_init, str):
            enc_dec = 'quiet' if weight_init == 'resonant' else weight_init
            return [enc_dec, weight_init, 'quiet', 'zero']

        return self._normalize_component_list(weight_init, defaults, 'weight_init')

    def _normalize_activation(self, activation):
        defaults = ['none', 'tanh', 'tanh', 'none']
        if activation is None:
            return defaults.copy()

        if isinstance(activation, str):
            normalized = defaults.copy()
            normalized[1] = activation
            return normalized

        return self._normalize_component_list(activation, defaults, 'activation')

    def _normalize_gate(self, gate):
        defaults = ['none', 'none', 'identity']
        if gate is None:
            return defaults.copy()

        if isinstance(gate, str):
            return [gate, gate, gate]

        return self._normalize_component_list(gate, defaults, 'gate')

    def _build_gate_activation(self, gate_name):
        if gate_name is None:
            return None
        if not isinstance(gate_name, str):
            raise TypeError(f"gate entries must be strings or None, got {type(gate_name).__name__}")

        if gate_name.lower() == 'none':
            return None

        return self._build_activation(gate_name)

    def _create_gate_parameter(self, dim, gate_act, device):
        if gate_act is None:
            return None
        return nn.Parameter(torch.empty(dim, device=device))

    def _get_input_scale(self, dtype):
        input_scale = self.input_scale.to(dtype)
        if self.enc_dec_gate_act is not None and self.input_gate is not None:
            input_scale = input_scale * self.enc_dec_gate_act(self.input_gate.to(dtype))
        return input_scale

    def _get_output_scale(self, dtype):
        output_scale = self.output_scale.to(dtype)
        if self.enc_dec_gate_act is not None and self.output_gate is not None:
            output_scale = output_scale * self.enc_dec_gate_act(self.output_gate.to(dtype))
        return output_scale

    def _init_weights(self):
        self._apply_init(self.W, self.core_weight_init)
        
        if self.embed is not None:
            self._apply_init(self.embed.weight, self.enc_dec_weight_init)
        if self.proj is not None:
            self._apply_init(self.proj.weight, self.enc_dec_weight_init)
        if self.output_decoder is not None:
            self._apply_init(self.output_decoder.weight, self.enc_dec_weight_init)
        if self.input_gate is not None:
            self._apply_init(self.input_gate, self.gate_weight_init)
        if self.output_gate is not None:
            self._apply_init(self.output_gate, self.gate_weight_init)
        if self.core_gate is not None:
            self._apply_init(self.core_gate, self.gate_weight_init)
        if self.memory_gate is not None:
            self._apply_init(self.memory_gate, self.gate_weight_init)
        
    def _apply_init(self, tensor, strategy):
        """
        Applies requested weight initialization strategy to a specific tensor.
        """
        with torch.no_grad():
            if strategy == 'quiet':
                nn.init.normal_(tensor, mean=0.0, std=0.02)
            elif strategy == 'micro_quiet':
                nn.init.normal_(tensor, mean=0.0, std=1e-6)
            elif strategy == 'micro_quiet_warm':
                nn.init.normal_(tensor, mean=0.0, std=1e-3)
            elif strategy == 'classic':
                nn.init.normal_(tensor)
            elif strategy == 'xavier_uniform':
                nn.init.xavier_uniform_(tensor)
            elif strategy == 'xavier_normal':
                nn.init.xavier_normal_(tensor)
            elif strategy == 'kaiming_uniform':
                nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
            elif strategy == 'kaiming_normal':
                nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
            elif strategy == 'orthogonal':
                nn.init.orthogonal_(tensor)
            elif strategy == 'sparse':
                nn.init.sparse_(tensor, sparsity=0.9, std=0.02)
            elif strategy == 'zero':
                nn.init.zeros_(tensor)
            elif strategy == 'one':
                nn.init.ones_(tensor)
            elif strategy == 'resonant':
                shape = tensor.shape
                
                signs = torch.randint(0, 2, shape, device=tensor.device).float() * 2.0 - 1.0
                noise = torch.randn(shape, device=tensor.device) * 0.02
                tensor.copy_(signs + noise)
                
                if hasattr(self, 'W') and tensor is self.W and tensor.ndim == 2:
                    tensor.fill_diagonal_(0.0)
                
                if tensor.ndim >= 2:
                    mat = tensor.view(tensor.shape[0], -1)
                    try:
                        sigma_max = torch.linalg.matrix_norm(mat, ord=2)
                        if sigma_max > 1e-8:
                            tensor.div_(sigma_max)
                    except (RuntimeError, ValueError) as e:
                        # Fallback to Frobenius norm if spectral norm fails
                        import warnings
                        warnings.warn(f"Spectral norm computation failed: {e}. Falling back to Frobenius norm.")
                        frob = tensor.norm()
                        if frob > 1e-8:
                            tensor.div_(frob / (tensor.numel() ** 0.5))
            else:
                nn.init.uniform_(tensor, -0.1, 0.1)

    def regenerate_weak_weights(self, threshold=0.01, percentage=None):
        with torch.no_grad():
            current_threshold = threshold
            if percentage is not None:
                current_threshold = torch.quantile(torch.abs(self.W), percentage).item()

            fresh_W = torch.empty_like(self.W)
            self._apply_init(fresh_W, self.weight_init_strategy)
            
            weak_mask = torch.abs(self.W) < current_threshold
            weak_mask.fill_diagonal_(False)
            
            count = weak_mask.sum().item()
            if count > 0:
                self.W.data[weak_mask] = fresh_W[weak_mask]
                # Revived synapses must start with a clean Hebbian state —
                # stale correlations from dead pathways would corrupt plasticity.
                if self.hebb_type is not None:
                    self.hebb_state_W[weak_mask] = 0.0
                if self.hebb_type == "synapse":
                    self.hebb_factor.data[weak_mask] = -3.0
                    self.hebb_decay.data[weak_mask]  =  2.2

            total_revived = count
            total_params = self.get_num_params()
            
            return total_revived, total_params
            
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if hasattr(self, 'W'):
            total -= self.W.shape[0]
        return total

    def compile(self):
        """
        Compiles the model using PyTorch 2.0 torch.compile for faster execution.
        Returns the compiled model (in-place modification where possible).
        """
        if hasattr(torch, 'compile'):
            try:
                # Use 'inductor' backend
                compiled_model = torch.compile(self)
                
                # FORCE DRY RUN to catch lazy errors now
                print("OdyssNet: Performing dry run to verify compilation...")
                if self.vocab_size is not None:
                    if self.embed is not None:
                        dummy_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    elif self.proj is not None:
                        dummy_input = torch.zeros(1, 1, self.proj.in_features, dtype=self.W.dtype, device=self.device)
                    else:
                        dummy_input = torch.zeros(1, self.num_neurons, device=self.device)
                else:
                    dummy_input = torch.zeros(1, self.num_neurons, device=self.device)
                with torch.no_grad():
                    compiled_model(dummy_input, steps=1)
                print("OdyssNet: Compilation successful!")
                return compiled_model
            except Exception as e:
                print(f"OdyssNet: Compilation failed ({e}). Fallback to eager execution.")
                return self
        else:
            print("OdyssNet: torch.compile not found. Skipping compilation.")
            return self

    def forward(self, x_input, steps=1, current_state=None, return_sequence=True):
        """
        Runs the dynamic system for `steps` timesteps.
        """
        if current_state is None:
            batch_sz = x_input.shape[0] if x_input is not None else 1
            if self.state.shape[0] != batch_sz:
                self.reset_state(batch_size=batch_sz)
            current_state = self.state
        else:
            batch_sz = current_state.shape[0]
        
        if current_state.device != self.device:
            current_state = current_state.to(self.device)

        h_t = current_state
        outputs = [] if return_sequence else None
        input_pos = cast(torch.Tensor, self.input_pos)
        output_pos = cast(torch.Tensor, self.output_pos)

        def _single_step(h_t_in, t_idx, x_input_info, hebb_W_contrib, hebb_mem_contrib):
            # hebb_W_contrib:   (N, N) tensor or None — added to W before recurrence.
            # hebb_mem_contrib: (N,)   tensor or None — added to memory_feedback.
            W_eff = self.W if hebb_W_contrib is None else self.W + hebb_W_contrib
            signal = h_t_in @ W_eff + self.B
            if self.debug: self._dbg(signal, f"signal/linear (step {t_idx})")

            if self.core_gate_act is not None and self.core_gate is not None:
                core_gate = self.core_gate_act(self.core_gate.to(signal.dtype))
                signal = signal * core_gate.unsqueeze(0)

            mem_weights = self.memory_feedback if hebb_mem_contrib is None else self.memory_feedback + hebb_mem_contrib
            feedback = h_t_in * mem_weights
            feedback = self.mem_act(feedback)
            if self.debug: self._dbg(feedback, f"memory_feedback (step {t_idx})")

            if self.mem_gate_act is not None and self.memory_gate is not None:
                memory_gate = self.mem_gate_act(self.memory_gate.to(feedback.dtype))
                feedback = feedback * memory_gate.unsqueeze(0)

            signal = signal + feedback
            if self.debug: self._dbg(signal, f"signal+feedback (step {t_idx})")

            input_scale = self._get_input_scale(signal.dtype)

            if x_input_info is not None:
                if isinstance(x_input_info, tuple):
                    if len(x_input_info) == 2 and x_input_info[0] is True:
                        # Out-of-place sparse injection for graph compiler compatibility
                        sparse_vec = cast(torch.Tensor, x_input_info[1])
                        signal = signal.index_add(1, input_pos, sparse_vec.to(signal.dtype))
                    elif len(x_input_info) == 3:
                        # Index-based Sparse Injection (Legacy)
                        v_mask = cast(torch.Tensor, x_input_info[0])
                        v_neurons = cast(torch.Tensor, x_input_info[1])
                        s_idx = cast(torch.Tensor, x_input_info[2])
                        if v_mask.any():
                            signal[v_mask, v_neurons] += input_scale[s_idx]
                else:
                    # Legacy Dense Injection
                    signal = signal + x_input_info

            if self.debug: self._dbg(signal, f"signal/pre-activation (step {t_idx})")
            activated = self.act(signal)
            if self.debug: self._dbg(activated, f"activated/{self.act.__class__.__name__} (step {t_idx})")

            # Dropout & StepNorm
            out = self.norm(self.drop(activated))
            if self.debug: self._dbg(out, f"step_norm output (step {t_idx})")
            return out

        # Thinking Ratio (Temporal Stretching)
        ratio = 1
        max_outputs = steps

        # Determine ratio for sequential inputs
        if x_input is not None:
            is_index_seq = x_input.dtype in [torch.long, torch.int64, torch.int32] and x_input.ndim == 2
            is_dense_seq = x_input.ndim == 3 and not self.pulse_mode

            if (is_index_seq or is_dense_seq) and x_input.shape[1] > 0:
                ratio = max(1, steps // x_input.shape[1])
                max_outputs = x_input.shape[1]

        # Initialise per-forward Hebbian state from the persisted buffer.
        # hebb_lr / hebb_ret are sigmoid-bounded tensors (scalar, vector, or matrix)
        # computed once per forward call; their shape matches hebb_type.
        if self.hebb_type is not None:
            hebb_lr  = torch.sigmoid(self.hebb_factor)
            hebb_ret = torch.sigmoid(self.hebb_decay)
            # Clone so that the end-of-loop copy_() back into the buffers does not
            # invalidate any saved tensors captured by the autograd graph.
            local_hebb_W   = self.hebb_state_W.detach().clone()
            local_hebb_mem = self.hebb_state_mem.detach().clone()

        # Precompute Dense Input/Output Scale Vectors
        input_scale_vec = torch.ones(self.num_neurons, dtype=h_t.dtype, device=self.device)
        if len(input_pos) > 0:
            input_scale_vec[input_pos] = self._get_input_scale(h_t.dtype)
        
        output_scale_vec = torch.ones(self.num_neurons, dtype=h_t.dtype, device=self.device)
        if len(output_pos) > 0:
            output_scale_vec[output_pos] = self._get_output_scale(h_t.dtype)

        for t in range(steps):
            # Prepare input for this step
            x_step_info = None

            if x_input is not None:
                # --- VOCAB MODE ---
                if self.vocab_size is not None:
                    # Determine current step index in the input sequence
                    seq_idx = t // ratio
                    is_active_step = (t % ratio == 0) and (seq_idx < x_input.shape[1])

                    if is_active_step:
                        # 1. Component Extraction
                        if x_input.ndim == 2:    # (Batch, Seq) — likely token indices
                            step_in = x_input[:, seq_idx]
                        elif x_input.ndim == 3:  # (Batch, Seq, Feat) — continuous features
                            step_in = x_input[:, seq_idx]
                        else:
                            # Fallback for pulse / single-step input
                            step_in = x_input

                        # 2. Projection (Embed or Linear)
                        vector = None

                        # Discrete (Int/Long) -> Embedding
                        if step_in.dtype in [torch.long, torch.int64, torch.int32]:
                            if self.embed is not None:
                                vector = self.embed(step_in.long())
                            elif self.proj is not None:
                                # Fallback: integer input in continuous mode — cast and project
                                vector = self.proj(step_in.float())

                        # Continuous (Float) -> Linear Projection
                        else:
                            if self.proj is not None:
                                vector = self.proj(step_in)
                            elif self.embed is not None:
                                # Fallback: float input in discrete mode — cast to long
                                vector = self.embed(step_in.long())

                        # 3. Map to Network State
                        if vector is not None:
                            vector = self.enc_dec_act(vector)
                            vector = vector * self._get_input_scale(vector.dtype)
                            # Sparse tuple payload: (Flag, Data)
                            x_step_info = (True, vector)

                        # Cache input for continuous (non-pulse) persistence across steps
                        if not self.pulse_mode:
                            self._cached_scaled_input = x_step_info

                    # Re-use cached input for non-active steps in continuous mode
                    if not self.pulse_mode and x_step_info is None:
                        x_step_info = getattr(self, '_cached_scaled_input', None)

                # --- LEGACY DIRECT MODE ---
                else:
                    # Handle Index-Based Input (VRAM Efficient)
                    if x_input.dtype in [torch.long, torch.int64, torch.int32]:
                        if x_input.ndim == 2:
                            if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                                token_indices = x_input[:, t // ratio]
                                valid_mask = token_indices != -1

                                if valid_mask.any():
                                    token_values = token_indices[valid_mask].long()
                                    input_dim = input_pos.numel()

                                    # Fast path: token values are local indices into input_ids.
                                    in_local_range = (token_values >= 0) & (token_values < input_dim)
                                    if in_local_range.all():
                                        scale_indices = token_values
                                        valid_neurons = input_pos[scale_indices]
                                        x_step_info = (valid_mask, valid_neurons, scale_indices)
                                    else:
                                        # Fallback: token values are explicit neuron IDs.
                                        if not hasattr(self, '_input_id_to_local'):
                                            self._input_id_to_local = {int(neuron_id): idx for idx, neuron_id in enumerate(self.input_ids)}

                                        active_batch_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
                                        mapped_batch = []
                                        mapped_local = []
                                        for b_idx, neuron_id in zip(active_batch_indices.tolist(), token_values.tolist()):
                                            local_idx = self._input_id_to_local.get(int(neuron_id))
                                            if local_idx is not None:
                                                mapped_batch.append(b_idx)
                                                mapped_local.append(local_idx)

                                        if mapped_local:
                                            sparse_mask = torch.zeros_like(valid_mask)
                                            sparse_mask[torch.tensor(mapped_batch, device=valid_mask.device)] = True
                                            scale_indices = torch.tensor(mapped_local, dtype=torch.long, device=token_values.device)
                                            valid_neurons = input_pos[scale_indices]
                                            x_step_info = (sparse_mask, valid_neurons, scale_indices)

                    elif x_input.ndim == 3:
                        # Sequential Input: (Batch, MultiSteps, Neurons)
                        if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                            x_step_info = x_input[:, t // ratio, :] * input_scale_vec

                    elif self.pulse_mode:
                        if t == 0:
                            x_step_info = x_input * input_scale_vec
                    else:
                        # Continuous mode: cache on first step, reuse for all subsequent steps
                        if t == 0:
                            self._cached_scaled_input = x_input * input_scale_vec
                        x_step_info = self._cached_scaled_input


            # Compute per-step Hebbian contributions before advancing the state.
            # cur_hebb_W and cur_hebb_mem carry gradients through hebb_lr (and hebb_ret
            # from step ≥ 1) because local_hebb_W accumulates hebb_lr * corr from the
            # previous step. hebb_W_contrib is passed explicitly to _single_step so that
            # gradient_checkpointing can correctly save and recompute it per step.
            if self.hebb_type is not None:
                h_prev = h_t
                if self.hebb_type == "neuron":
                    # hebb_lr: (N,) — broadcast as (N,1) over (N,N) to scale rows.
                    cur_hebb_W   = hebb_lr.unsqueeze(1) * local_hebb_W
                    cur_hebb_mem = hebb_lr * local_hebb_mem
                elif self.hebb_type == "synapse":
                    # hebb_lr: (N,N) — element-wise; diagonal used for memory path.
                    cur_hebb_W   = hebb_lr * local_hebb_W
                    cur_hebb_mem = hebb_lr.diagonal() * local_hebb_mem
                else:  # "global"
                    cur_hebb_W   = hebb_lr * local_hebb_W
                    cur_hebb_mem = hebb_lr * local_hebb_mem
                if self.debug: self._dbg(cur_hebb_W,   f"cur_hebb_W (step {t})")
                if self.debug: self._dbg(cur_hebb_mem, f"cur_hebb_mem (step {t})")
            else:
                cur_hebb_W   = None
                cur_hebb_mem = None

            # Gradient checkpointing
            if self.gradient_checkpointing and self.training:
                h_t = checkpoint.checkpoint(
                    _single_step, h_t, torch.tensor(t), x_step_info,
                    cur_hebb_W, cur_hebb_mem,
                    use_reentrant=False,
                )
            else:
                h_t = _single_step(h_t, t, x_step_info, cur_hebb_W, cur_hebb_mem)

            # Update local Hebbian state from temporal correlation h_t ⊗ h_{t-1}.
            # Old local state is detached to bound the computation graph to one step,
            # while hebb_lr and hebb_ret remain in the graph for gradient flow.
            if self.hebb_type is not None:
                batch_sz = h_t.size(0)
                # AMP autocast overrides explicit .float() casts for matmul-family ops
                # (einsum included), forcing them back to float16. Disable autocast here
                # so the correlation is always accumulated in float32.
                with torch.amp.autocast(device_type=h_t.device.type, enabled=False):
                    h_t_f    = h_t.detach().float()
                    h_prev_f = h_prev.detach().float()
                    if self.debug: self._dbg(h_t_f,    f"h_t pre-corr (step {t})")
                    if self.debug: self._dbg(h_prev_f, f"h_prev pre-corr (step {t})")
                    corr_W   = torch.einsum('bj,bi->ji', h_prev_f, h_t_f) / (batch_sz * self.num_neurons)
                    corr_mem = (h_t_f * h_prev_f).mean(dim=0)
                corr_W.fill_diagonal_(0.0)      # self-correlations go to hebb_state_mem
                if self.debug: self._dbg(corr_W,   f"corr_W (step {t})")
                if self.debug: self._dbg(corr_mem, f"corr_mem (step {t})")
                if self.hebb_type == "neuron":
                    local_hebb_W   = hebb_ret.unsqueeze(1) * local_hebb_W.detach()   + hebb_lr.unsqueeze(1) * corr_W
                    local_hebb_mem = hebb_ret * local_hebb_mem.detach() + hebb_lr * corr_mem
                elif self.hebb_type == "synapse":
                    local_hebb_W   = hebb_ret * local_hebb_W.detach()   + hebb_lr * corr_W
                    local_hebb_mem = hebb_ret.diagonal() * local_hebb_mem.detach() + hebb_lr.diagonal() * corr_mem
                else:  # "global"
                    local_hebb_W   = hebb_ret * local_hebb_W.detach()   + hebb_lr * corr_W
                    local_hebb_mem = hebb_ret * local_hebb_mem.detach() + hebb_lr * corr_mem
                if self.debug: self._dbg(local_hebb_W,   f"local_hebb_W (step {t})")
                if self.debug: self._dbg(local_hebb_mem, f"local_hebb_mem (step {t})")

            # Smart Output Collection
            if return_sequence and (t + 1) % ratio == 0 and len(outputs) < max_outputs:
                outputs.append(h_t)

        # Persist the recurrent state and Hebbian correlations for the next forward call.
        with torch.no_grad():
            self.state = h_t.detach()
            if self.hebb_type is not None:
                self.hebb_state_W.copy_(local_hebb_W.detach())
                self.hebb_state_mem.copy_(local_hebb_mem.detach())

        # Apply Output Scaling
        if return_sequence:
            if not outputs:
                stacked_outputs = h_t.unsqueeze(1)[:, :0, :]
            else:
                stacked_outputs = torch.stack(outputs, dim=1)
        else:
            # Avoid allocating (B, T, N) when only the final state is needed.
            stacked_outputs = h_t.unsqueeze(1)
            
        stacked_outputs = stacked_outputs * output_scale_vec.unsqueeze(0).unsqueeze(0)

        # Vocab Decoding
        if self.output_decoder is not None:
            # Extract only the output neurons
            out_activity = stacked_outputs[:, :, output_pos]
            # Project to Vocab
            # Shape: (Batch, Steps, OutNeurons) -> (Batch, Steps, Vocab)
            decoded = self.output_decoder(out_activity)
            decoded = self.enc_dec_act(decoded)
            return decoded, h_t

        return stacked_outputs, h_t

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)
        if self.hebb_type is not None:
            self.hebb_state_W.zero_()
            self.hebb_state_mem.zero_()

    def detach_state(self):
        """
        Detaches the internal state from the computational graph.
        Useful for Truncated BPTT.
        """
        self.state = self.state.detach()

    @property
    def device(self):
        return self.W.device
