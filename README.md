# LayerDiffuse with SDXL for Hugging Face Diffusers

This is a port of [Layer Diffuse](https://github.com/layerdiffusion/LayerDiffuse), from the [SD Forge extension](https://github.com/layerdiffusion/sd-forge-layerdiffuse) to Hugging Face Diffusers.

It focuses on SDXL and RGBA image generation.

## TL;DR

### Install requirements
```
pip install -r requirements.txt
```

### Run
```
python demo_sdxl_attn.py \
      --prompt "portrait of woman in suit with messy hair, high resolution, photorealistic, uniform textureless background" \
      --negative_prompt "ugly, bad, shadow, artifact, blurry"
```
![portrait of woman in suit with messy hair, high resolution, photorealistic, uniform textureless background](./examples/sdxl_layerdiffuse_00.png)

## Details

### Demo
``` python
from diffusers_extension.pipeline_stable_diffusion_xl_layer_diffuse import StableDiffusionXLLayerDiffusePipeline

pipeline = StableDiffusionXLLayerDiffusePipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

images = pipeline(
    prompt="portrait of woman in suit with messy hair, high resolution, photorealistic, uniform textureless background",
    negative_prompt="ugly, bad, shadow, artifact, blurry",
    num_inference_steps=20,
    width=1024,
    height=1024,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images

images[0].save("sdxl_layerdiffuse_result.png")

```

### Implementation


### Full arguments list
```
python demo_sdxl_attn.py \
      --seed SEED \
      --batch_size BATCH_SIZE \
      --guidance_scale GUIDANCE_SCALE \
      --num_inference_steps NUM_INFERENCE_STEPS \
      --width WIDTH \
      --height HEIGHT \
      --prompt PROMPT \
      --negative_prompt NEGATIVE_PROMPT \
      --output_path OUTPUT_PATH \
      --disable_memory_optim
```