import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import time
import random
import math
from tokenizers import Tokenizer, ByteLevelBPETokenizer
try:
    import tiktoken
except ImportError:
    pass

# --- ENVIRONMENT & IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, save_checkpoint, load_checkpoint, transplant_weights, ChaosGradConfig, TemporalSchedulerConfig
from datasets import load_dataset

# TF32 Optimization (Consistent with Notebook)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CONFIGURATION ---
TRUNCATED_BPTT_SEQ_LEN = 5
GENERATION_LENGTH = 128
SEQ_LEN = 5 if TRUNCATED_BPTT_SEQ_LEN == -1 else 128
BATCH_SIZE = -1
STEPS_PER_EPOCH = 10
LOG_INTERVAL = 1
MAX_START_SKIP = 1000
RESET_DATA_ITER = False
NUM_NEURONS = 512
INPUT_NEURON_COUNT = -1
OUTPUT_NEURON_COUNT = -1
ACTIVATION = 'gelu'
THINK_GAP = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NEUROGENESIS CONFIG
NEUROGENESIS_ENABLED = False
MAX_LOSS_INCREASE = 10
NEUROGENESIS_AMOUNT = 10

# REGENERATION CONFIG (PHOENIX)
DARWINIAN_REGENERATION = False
REGENERATION_MODE = 'percentage'
REGENERATION_THRESHOLD = 0.01
REGENERATION_PERCENTAGE = 0.001
REGENERATION_INTERVAL = 10

# TOKENIZER CONFIG
USE_TIKTOKEN = False
TIKTOKEN_ENCODING = "o200k_base"
CUSTOM_VOCAB_SIZE = 8192

# OPTIMIZER CONFIG
RESET_OPTIMIZER_ON_LOAD = False
OVERWRITE_LR_OF_CKPT = True
LEARNING_RATE = 1e-4

# TIE EMBEDDINGS (VRAM Saving & Parameter Sharing)
TIE_EMBEDDINGS = False

# SCHEDULER CONFIG (Now uses TemporalScheduler)
USE_SCHEDULER = False
WARMUP_STEPS = 500       # Linear warmup phase
MAX_STEPS = 5000         # Total steps for cosine decay to reach minimum
MIN_LR_RATIO = 0.01      # Decay down to 1% of max LR (i.e., 1e-6)

# --- TOKENIZER ---
def get_or_train_tokenizer():
    CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
    os.makedirs(CKPT_DIR, exist_ok=True)
    k_size = CUSTOM_VOCAB_SIZE // 1000
    TOKENIZER_PATH = os.path.join(CKPT_DIR, f"poc_tokenizer_{k_size}k.json")
    
    if os.path.exists(TOKENIZER_PATH):
        print(f"📚 Loading existing {k_size}k BPE Tokenizer from {TOKENIZER_PATH}...")
        return Tokenizer.from_file(TOKENIZER_PATH)
    
    print(f"📚 Training new {k_size}k BPE Tokenizer from data slice...")
    tokenizer = ByteLevelBPETokenizer()
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    texts = []
    count = 0
    for item in dataset:
        texts.append(item['text'])
        count += 1
        if count >= 100000: break
    
    tokenizer.train_from_iterator(texts, vocab_size=CUSTOM_VOCAB_SIZE, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    tokenizer.save(TOKENIZER_PATH)
    print(f"✅ Tokenizer saved to {TOKENIZER_PATH}")
    return tokenizer

class TiktokenWrapper:
    def __init__(self, encoding_name="o200k_base"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken is not installed. Please `pip install tiktoken` or set USE_TIKTOKEN = False.")
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.eos_id = self.tokenizer.eot_token
        
    def encode(self, text):
        class Encoded:
            def __init__(self, ids):
                self.ids = ids
        return Encoded(self.tokenizer.encode(text, allowed_special="all"))
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
    def get_vocab_size(self):
        return self.tokenizer.n_vocab
        
    def token_to_id(self, token):
        if token == "</s>" or token == "<|endoftext|>":
            return self.eos_id
        try:
            return self.tokenizer.encode(token, allowed_special="all")[0]
        except Exception:
            return self.eos_id

def get_tokenizer():
    if USE_TIKTOKEN:
        print(f"📚 Loading Tiktoken tokenizer: {TIKTOKEN_ENCODING}...")
        return TiktokenWrapper(TIKTOKEN_ENCODING)
    else:
        return get_or_train_tokenizer()

TOKENIZER = get_tokenizer()
VOCAB_SIZE = TOKENIZER.get_vocab_size()

# --- DATASET ---

class TinyStoriesIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, seq_len, tokenizer, skip_offset=0, debug=False):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.skip_offset = skip_offset
        self.debug = debug
        self.current_doc_index = skip_offset
        print("🌊 Connecting to the dataset...")
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def __iter__(self):
        start_skip = self.skip_offset

        if start_skip == 0 or RESET_DATA_ITER:
             start_skip = random.randint(0, MAX_START_SKIP)
             print(f"🔀 Random Start: Skipping {start_skip} documents...")
        else:
             print(f"⏩ Resuming from Document #{start_skip}...")

        # Worker-local index tracking
        local_doc_index = start_skip

        if start_skip > 0:
            iterator = iter(self.dataset.skip(start_skip))
        else:
            iterator = iter(self.dataset)

        buffer_tokens = []

        while True:
            # Replenish buffer
            while len(buffer_tokens) < self.seq_len + 1:
                try:
                    item = next(iterator)
                    local_doc_index += 1
                    if self.debug and local_doc_index % 1000 == 0:
                        print(f"📊 Streaming Index: Document #{local_doc_index}")
                    text = item.get('text', '')
                    encoded = self.tokenizer.encode(text)
                    buffer_tokens.extend(encoded.ids)
                    buffer_tokens.append(self.tokenizer.token_to_id("</s>"))
                except StopIteration:
                    iterator = iter(self.dataset)
                    local_doc_index = 0

            # Extract chunk
            chunk_tokens = buffer_tokens[:self.seq_len + 1]
            buffer_tokens = buffer_tokens[self.seq_len + 1:]

            indices = chunk_tokens

            if len(indices) == self.seq_len + 1:
                x = torch.tensor(indices[:-1], dtype=torch.long)
                y = torch.tensor(indices[1:], dtype=torch.long)
                # Yield index too so main process knows where we are
                yield x, y, local_doc_index

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

def generate(model, tokenizer, start_str="The", length=None, temperature=0.8, top_k=40, top_p=0.9):
    if length is None:
        length = GENERATION_LENGTH

    model.eval()

    encoded = tokenizer.encode(start_str)
    input_seq = encoded.ids

    current_state = None

    # Warm up state (Native Thinking)
    x_in = torch.tensor(input_seq, dtype=torch.long, device=model.device).unsqueeze(0)
    steps_total = x_in.shape[1] * (THINK_GAP + 1)
    
    with torch.no_grad():
        _, current_state = model(x_in, steps=steps_total)

        last_token_idx = input_seq[-1]

        for _ in range(length):
            # Native Single Step Generation
            total_step_single = 1 + THINK_GAP

            x_next = torch.tensor([[last_token_idx]], dtype=torch.long, device=model.device)

            preds, current_state = model(x_next, steps=total_step_single, current_state=current_state)
            
            logits = preds[0, 0, :]

            # Sampling logic
            if temperature > 0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, len(logits)))
                logits[logits < v[-1]] = float('-inf')

            if top_p is not None and top_p > 0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=0)

            if torch.isnan(probs).any() or torch.sum(probs) == 0:
                probs = torch.ones_like(probs) / len(probs)

            next_idx = torch.multinomial(probs, 1).item()

            input_seq.append(next_idx)
            last_token_idx = next_idx

    return tokenizer.decode(input_seq)

def initialize_system(vocab_size, num_neurons, device, input_count=-1, output_count=-1, lr=1e-4, activation='gelu'):
    if input_count == -1:
        input_neuron_count = num_neurons // 2
    else:
        input_neuron_count = input_count

    if output_count == -1:
        output_neuron_count = num_neurons // 2
    else:
        output_neuron_count = output_count
    
    input_ids = list(range(input_neuron_count))
    output_ids = list(range(input_neuron_count, input_neuron_count + output_neuron_count))
    
    # If num_neurons is -1, auto-size will handle it based on max(output_ids)
    
    model = RealNet(
        num_neurons=num_neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=device,
        dropout_rate=0.0,
        activation=activation,
        weight_init='orthogonal',
        gradient_checkpointing=True,
        vocab_size=vocab_size,
        vocab_mode='discrete',
        tie_embeddings=TIE_EMBEDDINGS
    )

    # Build scheduler config from global settings
    sched_config = None
    if USE_SCHEDULER:
        sched_config = dict(
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            min_lr_ratio=MIN_LR_RATIO,
            patience=200,
            cooldown=100,
            auto_extend=True,
        )

    trainer = RealNetTrainer(
        model, lr=lr, device=device,
        gradient_persistence=0.0, synaptic_noise=0,
        chaos_config=ChaosGradConfig.aggressive(lr=lr),
        scheduler_config=sched_config,
    )

    return model, trainer, input_ids, output_ids

def calculate_optimal_batch_size(model, device, seq_len, think_gap, truncated_bptt_seq_len):
    """Calculates optimal batch size based on VRAM capacity."""
    print("\n⚖️  Auto-Tuning Batch Size...")

    if device == 'cpu':
        return 32

    if device == 'cuda':
        t = torch.cuda.get_device_properties(0).total_memory
        a = torch.cuda.memory_allocated(0)
        
        # 1. BASELINE VRAM (Batch-size independent)
        # Note: 'actual_total_params' naturally reflects True param count. If tie_embeddings is enabled,
        # the parameter count is automatically smaller, accurately reflecting the VRAM savings.
        actual_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Optimizer overhead + PyTorch/CUBLAS contexts (~1.5 GB workspace limit)
        baseline_vram = (actual_total_params * 16) + (1.5 * 1024 * 1024 * 1024)
        
        # 2. ACTIVATIONS VRAM PER SAMPLE (Batch-size dependent)
        num_neurons = model.num_neurons
        
        if hasattr(model, 'vocab_size') and model.vocab_size is not None:
             vocab_dim = model.vocab_size[-1] if isinstance(model.vocab_size, (list, tuple)) else model.vocab_size
        else:
             vocab_dim = num_neurons
             
        # Effective steps
        if truncated_bptt_seq_len > 0:
            effective_mem_len = truncated_bptt_seq_len * (think_gap + 1)
            logit_seq_len = truncated_bptt_seq_len
        else:
            effective_mem_len = seq_len * (think_gap + 1)
            logit_seq_len = seq_len

        # A: Core execution memory per sample 
        # Gradient Checkpointing saves input state per step (4 bytes) + Gradients (4 bytes) = 8 bytes
        core_mem_per_sample = effective_mem_len * num_neurons * 8
        
        # B: Vocab / Logits memory per sample
        # Logits (4) + Softmax Probs (4) + Gradients (4) = 12 bytes
        logit_mem_per_sample = logit_seq_len * vocab_dim * 12

        vram_per_sample = core_mem_per_sample + logit_mem_per_sample
        
        # CUDA context margin
        vram_per_sample = vram_per_sample * 1.1

        # We keep 90% of available free VRAM as safe margin
        free_vram = t - a - baseline_vram
        safe_vram = free_vram * 0.9
        
        calc_batch = int(safe_vram / vram_per_sample) if vram_per_sample > 0 else 1
        calc_batch = max(1, calc_batch)

        # Better hardware utilization if batch size is a multiple of 8 (Tensor Cores limit)
        if calc_batch > 8:
            calc_batch = (calc_batch // 8) * 8

        print(f"   VRAM Total:  {t / 1e9:.2f} GB")
        print(f"   Model Base:  {baseline_vram / 1e9:.2f} GB (Params + Optim)")
        print(f"   VRAM Free:   {free_vram / 1e9:.2f} GB")
        print(f"   Mem/Sample:  {vram_per_sample / 1e6:.2f} MB")
        print(f"   Opt. Batch:  {calc_batch}")

        return calc_batch
    return 32

def main():
    global NUM_NEURONS, BATCH_SIZE # Allow updating global config if needed

    print(f"🚀 RealNet-LLM (TinyStories Streaming) - NATIVE THINKING MODE")
    print(f"--- Configuration ---")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE} (Will Auto-Tune if -1)")
    print(f"NUM_NEURONS: {NUM_NEURONS}")
    print(f"INPUT_NEURON_COUNT: {INPUT_NEURON_COUNT} (Half of NumNeurons if -1)")
    print(f"OUTPUT_NEURON_COUNT: {OUTPUT_NEURON_COUNT} (Half of NumNeurons if -1)")
    print(f"TRUNCATED_BPTT_SEQ_LEN (Tokens): {TRUNCATED_BPTT_SEQ_LEN}")
    print(f"STEPS_PER_EPOCH: {STEPS_PER_EPOCH}")
    print(f"LOG_INTERVAL: {LOG_INTERVAL}")
    print(f"MAX_START_SKIP: {MAX_START_SKIP}")
    print(f"RESET_DATA_ITER: {RESET_DATA_ITER}")
    print(f"GENERATION_LENGTH: {GENERATION_LENGTH}")
    print(f"THINK_GAP: {THINK_GAP}")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"TIE_EMBEDDINGS: {TIE_EMBEDDINGS}")
    tiktoken_info = f", Encoding: {TIKTOKEN_ENCODING}" if USE_TIKTOKEN else ""
    print(f"VOCAB_SIZE: {VOCAB_SIZE} (Tiktoken: {USE_TIKTOKEN}{tiktoken_info})")
    print(f"DEVICE: {DEVICE}")
    print(f"NEUROGENESIS: Enabled={NEUROGENESIS_ENABLED}, MaxLossInc={MAX_LOSS_INCREASE}, Amount={NEUROGENESIS_AMOUNT}")
    if DARWINIAN_REGENERATION:
        regen_val = f"{REGENERATION_PERCENTAGE:.2%}" if REGENERATION_MODE == 'percentage' else f"{REGENERATION_THRESHOLD}"
        print(f"PHOENIX (Regeneration): Mode={REGENERATION_MODE}, Val={regen_val}, Interval={REGENERATION_INTERVAL}")
    else:
        print(f"PHOENIX (Regeneration): Disabled")
    print(f"RESET_OPTIM_ON_LOAD: {RESET_OPTIMIZER_ON_LOAD}")
    print(f"SCHEDULER: Enabled={USE_SCHEDULER}, Warmup={WARMUP_STEPS}, MaxSteps={MAX_STEPS}, MinRatio={MIN_LR_RATIO}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"OVERWRITE_LR_OF_CKPT: {OVERWRITE_LR_OF_CKPT}")
    print(f"---------------------")

    # --- CHECKPOINT PRE-LOAD (For Data Resume) ---
    CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(CKPT_DIR, f'llm_tinystories_{ACTIVATION}_latest.pth')
    CKPT_BEST_PATH = os.path.join(CKPT_DIR, f'llm_tinystories_{ACTIVATION}_best.pth')

    resume_doc_index = 0
    start_epoch = 0

    # --- CHECKPOINT PRE-LOAD (Dataset Resume) ---
    if os.path.exists(CKPT_PATH):
        try:
             peek = torch.load(CKPT_PATH, map_location='cpu')
             resume_doc_index = peek.get('dataset_step', 0)
             if resume_doc_index > 0:
                 print(f"📂 Resuming dataset from index: {resume_doc_index}")

             start_epoch = peek.get('epoch', -1) + 1
        except:
             pass

    dataset = TinyStoriesIterableDataset(SEQ_LEN, TOKENIZER, skip_offset=resume_doc_index, debug=False)

    # --- MODEL SETUP ---
    model, trainer, input_ids, output_ids = initialize_system(VOCAB_SIZE, NUM_NEURONS, DEVICE, input_count=INPUT_NEURON_COUNT, output_count=OUTPUT_NEURON_COUNT, lr=LEARNING_RATE, activation=ACTIVATION)
    NUM_NEURONS = model.num_neurons

    print(f"Input IDs: {input_ids[0]}-{input_ids[-1]}")
    print(f"Output IDs: {output_ids[0]}-{output_ids[-1]}")

    # --- CHECKPOINT LOADING (Full) ---
    if os.path.exists(CKPT_PATH):
        # Pre-check dimensions to handle mismatches interactively
        try:
            ckpt_peek = torch.load(CKPT_PATH, map_location=DEVICE)
            if 'model_state_dict' in ckpt_peek and 'W' in ckpt_peek['model_state_dict']:
                saved_dim = ckpt_peek['model_state_dict']['W'].shape[0]

                if saved_dim != NUM_NEURONS:
                    print(f"\n⚠️ ARCHITECTURE MISMATCH DETECTED!")
                    print(f"   Current Model: {NUM_NEURONS}")
                    print(f"   Checkpoint:    {saved_dim}")

                    print("Select action:")
                    print("[1] Resize Model (Resume)")
                    print("[2] Transplant Weights (Adapt)")
                    print("[3] Start Fresh")
                    action = input("Choice [1/2/3]: ").strip()

                    if action == '1':
                        print(f"🔄 Resizing to {saved_dim}...")
                        NUM_NEURONS = saved_dim
                        model, trainer, _, _ = initialize_system(VOCAB_SIZE, NUM_NEURONS, DEVICE, input_count=INPUT_NEURON_COUNT, output_count=OUTPUT_NEURON_COUNT, lr=LEARNING_RATE, activation=ACTIVATION)
                        opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                        target_lr = LEARNING_RATE if OVERWRITE_LR_OF_CKPT else None
                        load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True, lr=target_lr)
                        print(f"✅ Resuming from Epoch {start_epoch}")

                    elif action == '2':
                        print(f"⚠️ Transplanting Weights...")
                        transplant_weights(model, CKPT_PATH, device=DEVICE)
                        print(f"🧬 Transplant complete.")

                    else:
                        print("🆕 Starting fresh.")
                        start_epoch = 0
                        dataset.skip_offset = 0

                else:
                    opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                    target_lr = LEARNING_RATE if OVERWRITE_LR_OF_CKPT else None
                    print(f"🔄 Loading Checkpoint from {CKPT_PATH}...")
                    load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True, lr=target_lr)
                    print(f"✅ Resuming from Epoch {start_epoch}")
            else:
                 opt_arg = None if RESET_OPTIMIZER_ON_LOAD else trainer.optimizer
                 target_lr = LEARNING_RATE if OVERWRITE_LR_OF_CKPT else None
                 load_checkpoint(model, opt_arg, CKPT_PATH, device=DEVICE, strict=True, lr=target_lr)
                 start_epoch = ckpt_peek.get('epoch', 0) + 1

        except Exception as e:
            print(f"⚠️ Failed to load/inspect checkpoint: {e}. Starting fresh.")

    # --- BATCH SIZE OPTIMIZATION ---
    if BATCH_SIZE == -1:
         BATCH_SIZE = calculate_optimal_batch_size(
             model,
             DEVICE,
             SEQ_LEN,
             THINK_GAP,
             TRUNCATED_BPTT_SEQ_LEN
         )

    # DataLoader for IterableDataset
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )

    # CrossEntropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    trainer.loss_fn = criterion

    # OUTPUT TRANSFORM: Flatten (Batch, Steps, Out) -> (N, Out)
    def flatten_logits(out):
        return out.reshape(-1, dataset.get_vocab_size())

    # --- MODEL INFO ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- MODEL INFO ---")
    print(f"Total Trainable Parameters: {total_params:,}")

    # --- INITIAL TESTS ---
    print("\n--- GENERATION PREVIEW ---")
    try:
        gen_text = generate(model, TOKENIZER, start_str="Once upon a time")
        print(f"Sample: {gen_text}\n")
    except Exception as e:
        print(f"Error: {e}")

    print("--- TRAINING LOOP ---")

    epoch = start_epoch
    prev_loss = float('inf')
    loss_increase_counter = 0
    best_loss = float('inf')

    if os.path.exists(CKPT_BEST_PATH):
        try:
            best_ckpt = torch.load(CKPT_BEST_PATH, map_location=DEVICE)
            best_loss = best_ckpt.get('loss', float('inf'))
            print(f"🏆 Historical Best Loss: {best_loss:.4f}")
        except:
            pass

    # NOTE: TemporalScheduler is now integrated into the trainer.
    # It auto-steps inside train_batch(). No manual scheduler needed.

    data_iterator = iter(dataloader)

    while True:
        total_loss = 0
        steps = 0
        start_time = time.time()

        for batch_idx in range(STEPS_PER_EPOCH):
            try:
                x, y, current_doc_tensor = next(data_iterator)
                current_doc = current_doc_tensor[-1].item()
                dataset.current_doc_index = current_doc
            except StopIteration:
                print("🔄 Restarting iterator...")
                data_iterator = iter(dataloader)
                x, y, current_doc_tensor = next(data_iterator)
                current_doc = current_doc_tensor[-1].item()
                dataset.current_doc_index = current_doc

            # Native Thinking Preparation
            # x is raw tokens: (Batch, SeqLen)
            # y is raw target: (Batch, SeqLen)
            # We calculate total steps including gaps:
            # 1 Step (Read) + N Steps (Think) = N+1 steps per token
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            y_flat = y.reshape(-1)
            
            seq_len = x.shape[1]
            total_thinking_steps = seq_len * (THINK_GAP + 1)

            if TRUNCATED_BPTT_SEQ_LEN != -1 and TRUNCATED_BPTT_SEQ_LEN > 0:
                current_state = None
                batch_loss = 0
                steps_count = 0
                
                chunk_len = TRUNCATED_BPTT_SEQ_LEN
                
                for t_start in range(0, seq_len, chunk_len):
                    t_end = min(t_start + chunk_len, seq_len)
                    
                    # Extract sequence chunk
                    x_chunk = x[:, t_start:t_end]
                    y_chunk_flat = y[:, t_start:t_end].reshape(-1)
                    
                    # Thinking steps for the current chunk
                    actual_tokens = t_end - t_start
                    chunk_thinking_steps = actual_tokens * (THINK_GAP + 1)

                    loss, current_state = trainer.train_batch(
                        x_chunk,
                        y_chunk_flat,
                        thinking_steps=chunk_thinking_steps,
                        full_sequence=True,
                        output_transform=flatten_logits,
                        initial_state=current_state,
                        return_state=True
                    )

                    current_state = current_state.detach()
                    batch_loss += loss
                    steps_count += 1

                loss = batch_loss / max(steps_count, 1)

            else:
                loss = trainer.train_batch(
                    x,
                    y_flat,
                    thinking_steps=total_thinking_steps,
                    full_sequence=True,
                    output_transform=flatten_logits
                )

            # LR Tracking (TemporalScheduler auto-steps inside train_batch)
            if trainer.scheduler:
                current_lr = trainer.scheduler.get_last_lr()[0]
            else:
                current_lr = trainer.optimizer.param_groups[0]['lr']

            total_loss += loss
            steps += 1

            if batch_idx % LOG_INTERVAL == 0:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                ppl = np.exp(loss_val)
                print(f"Epoch {epoch} | Batch {batch_idx} | Doc #{current_doc} | Loss {loss:.4f} | PPL {ppl:.2f} | LR {current_lr:.2e}")

        avg_loss = total_loss / steps
        avg_loss_val = avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss
        avg_ppl = np.exp(avg_loss_val)
        print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Avg PPL {avg_ppl:.2f} | Time: {time.time() - start_time:.1f}s")

        # --- PERIODIC GENERATION ---
        print("--- GENERATION ---")
        try:
            gen_text = generate(model, TOKENIZER, start_str="Once upon a time ")
            print(gen_text)
        except Exception as e:
            print(f"Generation Error: {e}")
        print("------------------")

        # --- CHECKPOINT SAVING ---
        ckpt_extra_data = {
            'initial_lr': trainer.initial_lr,
            'dataset_step': dataset.current_doc_index
        }

        save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_PATH, extra_data=ckpt_extra_data)
        print(f"💾 Checkpoint Saved: {CKPT_PATH} (Doc Index: {dataset.current_doc_index})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, trainer.optimizer, epoch, avg_loss, CKPT_BEST_PATH, extra_data=ckpt_extra_data)
            print(f"🏆 NEW RECORD! Saved: {CKPT_BEST_PATH} (Loss: {best_loss:.4f})")

        # --- NEUROGENESIS CONTROL ---
        if NEUROGENESIS_ENABLED:
            if avg_loss > prev_loss:
                loss_increase_counter += 1
                print(f"⚠️ Loss Increased ({loss_increase_counter}/{MAX_LOSS_INCREASE})")

            if loss_increase_counter >= MAX_LOSS_INCREASE:
                print(f"🧬 Expanding Network (Neurogenesis)...")
                trainer.expand(amount=NEUROGENESIS_AMOUNT)
                NUM_NEURONS = model.num_neurons
                loss_increase_counter = 0
                prev_loss = float('inf')

                if USE_SCHEDULER:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        trainer.optimizer, T_0=SCHEDULER_T0, eta_min=SCHEDULER_ETA_MIN)
            else:
                prev_loss = avg_loss

        # --- REGENERATION CONTROL (PHOENIX) ---
        if DARWINIAN_REGENERATION and epoch % REGENERATION_INTERVAL == 0:
            print(f"🔥 Phoenix Protocol: Checking for dead synapses...")

            p_arg = REGENERATION_PERCENTAGE if REGENERATION_MODE == 'percentage' else None
            t_arg = REGENERATION_THRESHOLD

            revived, total = trainer.regenerate_synapses(threshold=t_arg, percentage=p_arg)

            if revived > 0:
                print(f"🔥 Reborn: {revived}/{total} ({revived/total:.2%}) synapses regenerated.")
                prev_loss = float('inf')

        epoch += 1

if __name__ == "__main__":
    main()