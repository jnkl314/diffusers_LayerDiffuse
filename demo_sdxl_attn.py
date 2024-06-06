import argparse
import logging
import os

import torch

from diffusers_extension.pipeline_stable_diffusion_xl_layer_diffuse import StableDiffusionXLLayerDiffusePipeline


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=-1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-g', '--guidance_scale', type=float, default=5.0)
    parser.add_argument('-i', '--num_inference_steps', type=int, default=20)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('-p', '--prompt', type=str,
                        default="portrait of woman in suit with messy hair, high resolution, photorealistic, uniform textureless background")
    parser.add_argument('-n', '--negative_prompt', type=str,
                        default="ugly, bad, shadow, artifact, blurry")
    parser.add_argument('-o', '--output_path', type=str, default="./")
    parser.add_argument('--disable_memory_optim', action="store_true")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

    pipeline = StableDiffusionXLLayerDiffusePipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Move to device
    pipeline.to(device)

    # Enable options to reduce memory consumption
    if not args.disable_memory_optim:
        pipeline.enable_vae_tiling()
        pipeline.enable_vae_slicing()
        if torch.cuda.is_available():
            pipeline.enable_xformers_memory_efficient_attention()

    if not torch.cuda.is_available():
        pipeline.to(torch.float32)

    # Initialize generator
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

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
        logger.info(f"Write {output_file_path}")
        images[bs_idx].save(output_file_path)


if __name__ == "__main__":
    main(handle_args())
