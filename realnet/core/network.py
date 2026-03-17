import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import numpy as np

class RealNet(nn.Module):
    def __init__(self, num_neurons, input_ids, output_ids, pulse_mode=True, dropout_rate=0.1, device='cpu', weight_init='orthogonal', activation='tanh', gradient_checkpointing=False, vocab_size=None, vocab_mode='hybrid', tie_embeddings=False):
        super(RealNet, self).__init__()
        
        # Auto-size to unique input+output IDs
        if num_neurons == -1:
             unique_ids = set(input_ids) | set(output_ids)
             if len(unique_ids) > 0:
                 max_id = max(unique_ids)
                 num_neurons = max_id + 1
                 
                 difference = num_neurons - len(unique_ids)
                 if difference > 0:
                      print(f"ℹ️ RealNet Auto-Sizing: Sparse IDs detected. Created {num_neurons} neurons (covering Max ID {max_id}). Unconnected neurons: {difference}")
             else:
                 num_neurons = 0
        
        self.num_neurons = num_neurons

        self.input_ids = input_ids
        self.output_ids = output_ids
        
        # Vocab / Projection Mode
        self.vocab_size = vocab_size
        self.vocab_mode = vocab_mode
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
                    print("⚠️ Auto-Tying Warning: Weight tying is not supported for 'continuous' (Linear) vocab_mode due to transposed dimensions.")

        # Buffers for fast indexing
        self.register_buffer('input_pos', torch.tensor(input_ids, dtype=torch.long, device=device))
        self.register_buffer('output_pos', torch.tensor(output_ids, dtype=torch.long, device=device))
        
        # Scaling Parameters (Input/Output)
        self.input_scale = nn.Parameter(torch.full((len(input_ids),), 1.0, device=device))
        self.output_scale = nn.Parameter(torch.full((len(output_ids),), 1.0, device=device))
        
        self.pulse_mode = pulse_mode
        self.activation_type = activation
        self.gradient_checkpointing = gradient_checkpointing
        self._device = device # Private variable for property
        self.weight_init_strategy = weight_init
        
        # Weight Matrix (N x N)
        self.W = nn.Parameter(torch.empty(num_neurons, num_neurons, device=device))
        self._init_weights(weight_init)
        
        # Bias Vector
        self.B = nn.Parameter(torch.zeros(num_neurons, device=device))

        # StepNorm (RMSNorm - faster than LayerNorm, used in modern LLMs)
        self.norm = nn.RMSNorm(num_neurons).to(device)
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
             self.act = nn.SiLU()
        else:
             raise ValueError(f"Unknown activation function: {activation}")

        self.drop = nn.Dropout(p=dropout_rate)

        # Internal State (hidden state h_t)
        self.state = torch.zeros(1, num_neurons, device=device)
        
    def _init_weights(self, strategy):
        # Core Matrix
        self._apply_init(self.W, strategy)
        
        # Projections & Embeddings (LLM Standards)
        if self.embed is not None:
            self._apply_init(self.embed.weight, strategy)
        if self.proj is not None:
            self._apply_init(self.proj.weight, strategy)
        if self.output_decoder is not None:
            self._apply_init(self.output_decoder.weight, strategy)
        
    def _apply_init(self, tensor, strategy):
        """
        Applies requested weight initialization strategy to a specific tensor.
        """
        with torch.no_grad():
            if strategy == 'quiet':
                nn.init.normal_(tensor, mean=0.0, std=0.02)
            elif strategy == 'micro_quiet':
                nn.init.normal_(tensor, mean=0.0, std=1e-6)
            elif strategy == 'micro_quiet_8bit':
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
            else:
                nn.init.uniform_(tensor, -0.1, 0.1)

    def regenerate_weak_weights(self, threshold=0.01, percentage=None):
        """
        Darwinian Regeneration (Phoenix Protocol).
        Re-initializes weights based on magnitude.
        
        Args:
            threshold (float): Absolute value usage threshold. (Used if percentage is None)
            percentage (float): If provided (0.0 < p < 1.0), regenerates the bottom p% of weights.
                                E.g., 0.05 will regenerate the weakest 5% of connections.
        """
        with torch.no_grad():
            # Determine Threshold dynamically if percentage is active
            current_threshold = threshold
            if percentage is not None:
                # Calculate the quantile value (e.g. the value at 10%)
                # We use abs() because we care about magnitude strength
                current_threshold = torch.quantile(torch.abs(self.W), percentage).item()

            # Create a full fresh tensor
            fresh_W = torch.empty_like(self.W)
            self._apply_init(fresh_W, self.weight_init_strategy)
            
            # Find weak spots
            weak_mask = torch.abs(self.W) < current_threshold
            
            # Transplant fresh cells into weak spots
            count = weak_mask.sum().item()
            if count > 0:
                self.W.data[weak_mask] = fresh_W[weak_mask]
            
            total_revived = count
            total_params = self.W.numel()
            
            return total_revived, total_params

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
                print("RealNet: Performing dry run to verify compilation...")
                dummy_input = torch.zeros(1, self.num_neurons, device=self.device)
                with torch.no_grad():
                    compiled_model(dummy_input, steps=1)
                print("RealNet: Compilation successful!")
                return compiled_model
            except Exception as e:
                print(f"RealNet: Compilation failed ({e}). Fallback to eager execution.")
                return self
        else:
            print("RealNet: torch.compile not found. Skipping compilation.")
            return self

    def forward(self, x_input, steps=1, current_state=None):
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
        outputs = []

        def _single_step(h_t_in, t_idx, x_input_info):
            # Projection
            signal = F.linear(h_t_in, self.W.t(), self.B)
            
            # Input Injection
            if x_input_info is not None:
                if isinstance(x_input_info, tuple):
                    if len(x_input_info) == 2 and x_input_info[0] is True:
                        # Out-of-place sparse injection for graph compiler compatibility
                        signal = signal.index_add(1, self.input_pos, x_input_info[1].to(signal.dtype))
                    else:
                        # Index-based Sparse Injection (Legacy)
                        v_mask, v_neurons, s_idx = x_input_info
                        if v_mask.any():
                            signal[v_mask, v_neurons] += self.input_scale[s_idx].to(signal.dtype)
                else:
                    # Legacy Dense Injection
                    signal = signal + x_input_info
            
            activated = self.act(signal)
            
            # Dropout & StepNorm
            return self.norm(self.drop(activated))

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

        for t in range(steps):
            # Prepare input for this step
            x_step_info = None
            
            if x_input is not None:
                # --- VOCAB MODE ---
                if self.vocab_size is not None:
                     x_step_info = None
                     
                     # Determine current step index in the input sequence
                     seq_idx = t // ratio
                     is_active_step = (t % ratio == 0) and (seq_idx < x_input.shape[1])
                     
                     if is_active_step:
                         # 1. Component Extraction
                         if x_input.ndim == 2: # (Batch, Seq) - likely indices
                             step_in = x_input[:, seq_idx]
                         elif x_input.ndim == 3: # (Batch, Seq, Feat) - continuous
                             step_in = x_input[:, seq_idx]
                         else:
                             # Fallback for Pulse/Single
                             step_in = x_input
                             
                         # 2. Projection (Embed or Linear)
                         vector = None
                         
                         # Discrete (Int/Long) -> Embedding
                         if step_in.dtype in [torch.long, torch.int64, torch.int32]:
                             if self.embed is not None:
                                 vector = self.embed(step_in.long()) # Ensure Long for embedding
                             else:
                                 # Fallback for integer inputs in continuous mode
                                 # Cast to float and project
                                 if self.proj is not None:
                                     vector = self.proj(step_in.float())
                                     
                         # Continuous (Float) -> Projection
                         else:
                             if self.proj is not None:
                                 vector = self.proj(step_in)
                             else:
                                 # Fallback for float inputs in discrete mode
                                 # Attempt to cast to long for embedding lookup 
                                 if self.embed is not None:
                                     vector = self.embed(step_in.long())
                         
                         # 3. Map to Network State
                         if vector is not None:
                              # Apply Input Scaling
                              vector = vector * self.input_scale.to(vector.dtype)
                              
                              # Sparse tuple payload: (Flag, Data)
                              x_step_info = (True, vector)
                         
                         # Caching for Continuous/Pulse persistence if needed
                         if self.pulse_mode and t==0:
                             pass # Done
                         elif not self.pulse_mode:
                              self._cached_scaled_input = x_step_info

                     # Handle persistence for continuous mode (non-pulse)
                     if not self.pulse_mode: 
                          # Re-use cached input if available (for static vocab inputs)
                          if x_step_info is None and hasattr(self, '_cached_scaled_input'):
                               x_step_info = self._cached_scaled_input

                # --- LEGACY DIRECT MODE ---
                else:
                    # Handle Index-Based Input (VRAM Efficient)
                    if x_input.dtype in [torch.long, torch.int64, torch.int32]:
                         if x_input.ndim == 2:
                              if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                                   token_indices = x_input[:, t // ratio]
                                   valid_mask = token_indices != -1
                                   
                                   if valid_mask.any():
                                        # Map token indices to neuron indices
                                        offset = self.input_ids[0]
                                        valid_neurons = token_indices[valid_mask] + offset
                                        scale_indices = valid_neurons - offset
                                        # Sparse info tuple
                                        x_step_info = (valid_mask, valid_neurons, scale_indices)
                                   
                    elif x_input.ndim == 3:
                        # Sequential Input: (Batch, MultiSteps, Neurons)
                        if t % ratio == 0 and (t // ratio) < x_input.shape[1]:
                            x_step = x_input[:, t // ratio, :].clone()
                            x_step[:, self.input_pos] = x_step[:, self.input_pos] * self.input_scale.to(x_step.dtype)
                            x_step_info = x_step
                            
                    elif self.pulse_mode:
                        if t == 0:
                            x_step = x_input.clone()
                            x_step[:, self.input_pos] = x_step[:, self.input_pos] * self.input_scale.to(x_step.dtype)
                            x_step_info = x_step
                    else:
                        # Continuous mode
                        if t == 0:
                            self._cached_scaled_input = x_input.clone()
                            self._cached_scaled_input[:, self.input_pos] = self._cached_scaled_input[:, self.input_pos] * self.input_scale.to(self._cached_scaled_input.dtype)
                        x_step_info = self._cached_scaled_input
            
            # Gradient checkpointing
            if self.gradient_checkpointing and self.training:
                h_t = checkpoint.checkpoint(_single_step, h_t, torch.tensor(t), x_step_info, use_reentrant=False)
            else:
                h_t = _single_step(h_t, t, x_step_info)
            
            # Smart Output Collection
            if (t + 1) % ratio == 0 and len(outputs) < max_outputs:
                outputs.append(h_t)

        # Apply Output Scaling
        stacked_outputs = torch.stack(outputs, dim=1)
        stacked_outputs[:, :, self.output_pos] = stacked_outputs[:, :, self.output_pos] * self.output_scale.to(stacked_outputs.dtype)
        
        # Vocab Decoding
        if self.output_decoder is not None:
             # Extract only the output neurons
             out_activity = stacked_outputs[:, :, self.output_pos]
             # Project to Vocab
             # Shape: (Batch, Steps, OutNeurons) -> (Batch, Steps, Vocab)
             decoded = self.output_decoder(out_activity)
             return decoded, h_t

        return stacked_outputs, h_t

    def reset_state(self, batch_size=1):
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.device)

    def detach_state(self):
        """
        Detaches the internal state from the computational graph.
        Useful for Truncated BPTT.
        """
        self.state = self.state.detach()

    @property
    def device(self):
        return self.W.device
