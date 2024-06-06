import argparse
import os

import torch

from diffusers_extension.pipeline_stable_diffusion_xl_layer_diffuse import StableDiffusionXLLayerDiffusePipeline
from diffusers_extension.utils import load_file_from_url


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=-1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-g', '--guidance_scale', type=float, default=5.0)
    parser.add_argument('-i', '--num_inference_steps', type=int, default=20)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('-p', '--prompt', type=str, default="an apple, high resolution")
    parser.add_argument('-n', '--negative_prompt', type=str, default="ugly, bad, shadow")
    parser.add_argument('-o', '--output_path', type=str, default="./")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

    # pipeline = StableDiffusionXLLayerDiffusePipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    # )
    pipeline = StableDiffusionXLLayerDiffusePipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v6",
        torch_dtype=torch.float16,
    )
    # sdxl_finetune_local_path = load_file_from_url(
    #     "https://huggingface.co/bluepen5805/anima_pencil-XL/resolve/main/anima_pencil-XL-v1.0.0.safetensors?download=true"
    # )
    # pipeline = StableDiffusionXLLayerDiffusePipeline.from_single_file(sdxl_finetune_local_path)   

    # Move to device
    pipeline.to(device)

    # Enable options to reduce memory consumption
    pipeline.enable_vae_tiling()
    pipeline.enable_vae_slicing()
    if torch.cuda.is_available():
        pipeline.enable_xformers_memory_efficient_attention()

    if not torch.cuda.is_available():
        pipeline.to(torch.float32)

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(12345)

    # Inference through pipeline, including patched SDXL VAE with LayerDiffuse + LayerDiffuse's UNet
    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        generator=gen,
        output_type="pil",
        num_images_per_prompt=args.batch_size
    ).images

    # Save image(s)
    for bs_idx in range(args.batch_size):
        output_file_path = os.path.join(args.output_path, f"sdxl_layerdiffuse_{bs_idx:02}.png")
        print(f"Write {output_file_path}")
        images[bs_idx].save(output_file_path)


if __name__ == "__main__":
    main(handle_args())
