import os
os.environ['HF_HOME'] = '/workspace/hf_cache'
import time
import json
import math
import random
import glob
import torch
import torch.nn.functional as F
from diffusers import Krea2Transformer2DModel
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Configurations
TOTAL_STEPS = 400
CHECKPOINT_INTERVAL = 100
LEARNING_RATE = 1e-4
MIN_LR = 1e-5
WARMUP_STEPS = 30
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ['to_q', 'to_k', 'to_v', 'to_out.0']

CACHE_DIR = '/workspace/dataset_cache'
SNAPSHOT_DIR = '/workspace/hf_cache/hub/models--krea--Krea-2-Raw/snapshots/6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603'
OUTPUT_DIR = '/workspace/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset into memory
print('Loading pre-cached dataset into memory...', flush=True)
latent_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*_latent.pt')))
dataset = []
for lat_f in latent_files:
    base = lat_f.replace('_latent.pt', '')
    txt_f = f'{base}_text.pt'
    lat = torch.load(lat_f, weights_only=True)
    txt = torch.load(txt_f, weights_only=True)
    name = os.path.basename(base)
    dataset.append({
        'name': name,
        'latent': lat,
        'prompt_embeds': txt['prompt_embeds'],
        'prompt_embeds_mask': txt['prompt_embeds_mask']
    })

position_ids = torch.load(os.path.join(CACHE_DIR, 'position_ids.pt'), weights_only=True).to('cuda')
print(f'Successfully loaded {len(dataset)} items into memory.', flush=True)

# 2. Load Krea 2 Transformer
print('Loading Krea 2 Transformer (12.8B) in bfloat16...', flush=True)
transformer = Krea2Transformer2DModel.from_pretrained(
    SNAPSHOT_DIR, subfolder='transformer', torch_dtype=torch.bfloat16
).to('cuda')
transformer.enable_gradient_checkpointing()

# 3. Inject PEFT LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias='none'
)
transformer = get_peft_model(transformer, lora_config)
transformer.train()

trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in transformer.parameters())
print(f'LoRA Injected: {trainable_params:,} trainable / {all_params:,} total ({100 * trainable_params / all_params:.3f}%)', flush=True)

# 4. Optimizer & LR Scheduler
optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * float(step) / float(max(1, WARMUP_STEPS))
    progress = float(step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LEARNING_RATE - MIN_LR) * cosine_decay

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 5. Training Loop
print(f'=== Starting Training: {TOTAL_STEPS} Steps ===', flush=True)
start_time = time.time()
loss_history = []
smoothed_loss = None

data_indices = []

for step in range(1, TOTAL_STEPS + 1):
    step_start = time.time()
    
    if not data_indices:
        data_indices = list(range(len(dataset)))
        random.shuffle(data_indices)
    
    idx = data_indices.pop()
    item = dataset[idx]
    
    current_lr = get_lr(step)
    set_lr(optimizer, current_lr)
    
    # Prepare batch
    latents = item['latent'].to('cuda', dtype=torch.bfloat16)
    embeds = item['prompt_embeds'].to('cuda', dtype=torch.bfloat16)
    mask = item['prompt_embeds_mask'].to('cuda')
    
    # Flow Matching Noise & Sigma
    noise = torch.randn_like(latents)
    u = torch.rand(1, device='cuda', dtype=torch.bfloat16)
    sigma = u.view(1, 1, 1)
    noisy_latents = (sigma * noise + (1.0 - sigma) * latents).to(torch.bfloat16)
    target = (noise - latents).to(torch.bfloat16)
    timestep = u.expand(1).to(torch.bfloat16)
    
    optimizer.zero_grad()
    
    pred = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=embeds,
        timestep=timestep,
        position_ids=position_ids,
        encoder_attention_mask=mask,
        return_dict=False
    )[0]
    
    loss = F.mse_loss(pred.float(), target.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
    optimizer.step()
    
    loss_val = loss.item()
    loss_history.append(loss_val)
    if smoothed_loss is None:
        smoothed_loss = loss_val
    else:
        smoothed_loss = 0.95 * smoothed_loss + 0.05 * loss_val
        
    step_duration = time.time() - step_start
    elapsed_total = time.time() - start_time
    avg_step_time = elapsed_total / step
    remaining_steps = TOTAL_STEPS - step
    eta_seconds = remaining_steps * avg_step_time
    vram_used = round(torch.cuda.memory_allocated() / (1024**3), 2)
    
    # Progress record
    progress_info = {
        'step': step,
        'total_steps': TOTAL_STEPS,
        'loss': round(loss_val, 4),
        'smoothed_loss': round(smoothed_loss, 4),
        'lr': f'{current_lr:.2e}',
        'step_time_sec': round(step_duration, 2),
        'elapsed_min': round(elapsed_total / 60, 2),
        'eta_min': round(eta_seconds / 60, 2),
        'vram_gb': vram_used,
        'item_name': item['name']
    }
    
    with open('/workspace/progress.json', 'w') as f:
        json.dump(progress_info, f, indent=2)
        
    if step % 5 == 0 or step == 1:
        print(f'Step [{step:03d}/{TOTAL_STEPS}] | Loss: {loss_val:.4f} (Avg: {smoothed_loss:.4f}) | LR: {current_lr:.2e} | StepTime: {step_duration:.2f}s | ETA: {eta_seconds/60:.1f}m | VRAM: {vram_used}GB', flush=True)
        
    # Checkpoint saving
    if step % CHECKPOINT_INTERVAL == 0 or step == TOTAL_STEPS:
        ckpt_dir = os.path.join(OUTPUT_DIR, f'checkpoint-step-{step}')
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f'>>> Saving Checkpoint at Step {step} to {ckpt_dir}...', flush=True)
        transformer.save_pretrained(ckpt_dir)
        
        try:
            from safetensors.torch import save_file
            peft_weights = {k: v.cpu() for k, v in transformer.state_dict().items() if 'lora_' in k}
            save_file(peft_weights, os.path.join(ckpt_dir, f'sofia_krea2_step_{step}.safetensors'))
            save_file(peft_weights, os.path.join(OUTPUT_DIR, f'sofia_krea2_step_{step}.safetensors'))
        except Exception as e:
            print(f'Warning saving standalone safetensors: {e}', flush=True)
            
        ckpt_meta = {
            'checkpoint_step': step,
            'timestamp': time.time(),
            'loss': loss_val,
            'smoothed_loss': smoothed_loss,
            'files': os.listdir(ckpt_dir)
        }
        with open(os.path.join(OUTPUT_DIR, 'latest_checkpoint.json'), 'w') as f:
            json.dump(ckpt_meta, f, indent=2)
        print(f'>>> Checkpoint {step} successfully saved!', flush=True)

# 6. Final Save
final_dir = os.path.join(OUTPUT_DIR, 'sofia_krea2_final')
os.makedirs(final_dir, exist_ok=True)
transformer.save_pretrained(final_dir)
from safetensors.torch import save_file
peft_weights = {k: v.cpu() for k, v in transformer.state_dict().items() if 'lora_' in k}
save_file(peft_weights, os.path.join(OUTPUT_DIR, 'sofia_krea2_lora_final.safetensors'))
save_file(peft_weights, os.path.join(final_dir, 'sofia_krea2_lora_final.safetensors'))

with open('/workspace/TRAINING_DONE', 'w') as f:
    f.write(f'Training completed successfully at {time.ctime()}\nTotal time: {(time.time() - start_time)/60:.2f} minutes\n')

print('=== Training Finished Successfully! ===', flush=True)
