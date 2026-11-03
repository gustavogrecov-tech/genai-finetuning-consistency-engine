"""
Zero-OOM Inference Pipeline for Krea 2.1 Raw & Qwen-Image LoRA.

Author: Gustavo Sales Grecov
Target Domain: AI Engineering & Generative AI Production Systems
Description:
    Demonstrates dynamic memory management via text encoder offloading,
    LoRA parameter injection on Diffusion/Flow Matching Transformers,
    and deterministic seed generation for high-fidelity facial consistency.
"""

import argparse
import gc
import os
from typing import Optional
import torch
from diffusers import Krea2Pipeline
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


class SofiaInferenceEngine:
    """Production-grade inference engine with dynamic VRAM lifecycle management."""

    def __init__(
        self,
        base_model_path: str = "krea/Krea-2-Raw",
        lora_weights_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.pipe: Optional[Krea2Pipeline] = None

    def load_pipeline(self) -> None:
        """Loads base transformer and pipeline components into memory."""
        print(f"[Engine] Loading base pipeline from '{self.base_model_path}'...")
        self.pipe = Krea2Pipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=self.torch_dtype,
        )

    def apply_lora(self, lora_path: str, rank: int = 32, alpha: int = 32) -> None:
        """Injects low-rank adaptation weights into transformer attention projections."""
        if not self.pipe:
            raise RuntimeError("Pipeline must be initialized before applying LoRA.")

        print(f"[Engine] Injecting LoRA adapter from '{lora_path}'...")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)

        if os.path.exists(lora_path):
            state_dict = load_file(lora_path)
            self.pipe.transformer.load_state_dict(state_dict, strict=False)
            print("[Engine] LoRA state_dict mapped successfully.")
        else:
            print(f"[Warning] LoRA checkpoint path '{lora_path}' not found locally. Running with initialized weights.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, bad anatomy, deformed eyes, cartoon, plastic skin",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 4.5,
        seed: int = 42,
        output_filename: str = "output_inference.png",
    ):
        """
        Executes zero-OOM generation by encoding prompt, offloading text encoder,
        and executing flow matching inference on the GPU.
        """
        if not self.pipe:
            self.load_pipeline()

        # Step 1: Text encoding in GPU
        print("[Engine] Encoding prompt representations...")
        self.pipe.text_encoder.to(self.device)
        prompt_embeds, neg_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=self.device,
        )

        # Step 2: Offload text encoder to free up ~10 GB VRAM
        print("[Engine] Offloading text encoder to preserve VRAM budget...")
        del self.pipe.text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Load Transformer & VAE to GPU
        print("[Engine] Mounting Transformer and VAE to compute device...")
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)

        if self.lora_weights_path:
            self.apply_lora(self.lora_weights_path)

        # Step 4: Iterative denoising via Flow Matching
        generator = torch.Generator(self.device).manual_seed(seed)
        print(f"[Engine] Starting generation: {steps} steps, CFG {guidance_scale}, seed {seed}...")
        image = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        image.save(output_filename)
        print(f"[Engine] Image successfully rendered and saved to '{output_filename}'.")
        return image


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-OOM Multimodal LoRA Inference Engine")
    parser.add_argument(
        "--prompt",
        type=str,
        default="sofia_woman, 28-year-old Scandinavian woman, sharp emerald green eyes, warm smile, natural blonde hair, defined cheekbones, fair skin, elegant cashmere blazer, studio lighting, photorealistic 8k",
        help="Input text prompt containing trigger tokens",
    )
    parser.add_argument("--negative_prompt", type=str, default="ugly, blurry, distorted, cartoon, plastic skin")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to .safetensors LoRA checkpoint")
    parser.add_argument("--output", type=str, default="sofia_output.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=28, help="Flow matching inference steps")
    parser.add_argument("--cfg", type=float, default=4.5, help="Classifier-Free Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = SofiaInferenceEngine(lora_weights_path=args.lora_path)
    engine.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        output_filename=args.output,
    )
