import logging
import os
from typing import Optional, Union, List, Dict, Any, Tuple, Callable

import safetensors
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.image_processor import PipelineImageInput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection, \
    CLIPImageProcessor

from diffusers_extension.models import TransparentVAEDecoder
from diffusers_extension.utils import load_file_from_url


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StableDiffusionXLLayerDiffusePipeline(StableDiffusionXLPipeline):
    """
    This class derives from StableDiffusionXLPipeline (currently diffusers==0.28.1) and add the
    necessary patch + extra VAE to perform Transparent Layer Diffusion https://arxiv.org/abs/2402.17113
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

        self.layer_model_cache_directory = os.path.join(os.path.expanduser("~"), ".cache", "layer_model")
        os.makedirs(self.layer_model_cache_directory, exist_ok=True)

        self.patch_with_layer_xl_transparent_attn()

        self.vae_transparent_decoder = self.load_transparent_vae_decoder()

        self.scheduler = self.load_scheduler()

    # Override the `to` method to move vae_transparent_decoder along
    def to(self, device):
        self.vae_transparent_decoder.to(device)
        # Call parent's `to` method to handle nn.Module attributes if any
        return super(StableDiffusionXLLayerDiffusePipeline, self).to(device)

    def create_layer_xl_transparent_attn_for_diffusers(self) -> str:
        """
        - Download the checkpoint layer_xl_transparent_attn.safetensors from LayerDiffusion/layerdiffusion-v1
        - Convert the name of the layers from SD Forge convention to Diffusers'
        - Save the modified checkpoint locally
        :return: Path of the modified checkpoint
        """
        layer_xl_transparent_attn_model_expected_path = os.path.join(
            self.layer_model_cache_directory,
            "layer_xl_transparent_attn.safetensors"
        )

        new_layer_xl_transparent_attn_model_path = os.path.join(
            os.path.dirname(layer_xl_transparent_attn_model_expected_path),
            f"[diffusers_format]_{os.path.basename(layer_xl_transparent_attn_model_expected_path)}"
        )

        # Only download and convert model if it doesn't already exist locally
        if not os.path.exists(new_layer_xl_transparent_attn_model_path):
            layer_xl_transparent_attn_model_path = load_file_from_url(
                url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
                model_dir=self.layer_model_cache_directory,
                file_name="layer_xl_transparent_attn.safetensors",
            )
            assert layer_xl_transparent_attn_model_expected_path == layer_xl_transparent_attn_model_path

            layer_lora_model = safetensors.torch.load_file(layer_xl_transparent_attn_model_path)

            # Create a dict for all replacements to be performed between the layers names
            # of layer_xl_transparent_attn.safetensors (SD Forge) and Diffusers' expected layer names
            replacement = {
                "diffusion_model.": "unet.",
                "to_q.weight::lora::0": "to_q_lora.up.weight",
                "to_q.weight::lora::1": "to_q_lora.down.weight",
                "to_k.weight::lora::0": "to_k_lora.up.weight",
                "to_k.weight::lora::1": "to_k_lora.down.weight",
                "to_v.weight::lora::0": "to_v_lora.up.weight",
                "to_v.weight::lora::1": "to_v_lora.down.weight",
                "to_out.0.weight::lora::0": "to_out_lora.up.weight",
                "to_out.0.weight::lora::1": "to_out_lora.down.weight",
            }

            # input_blocks 4.1 5.1 7.1 8.1   (wrong mapping: 4.1 5.1 7.1 8.1)
            # down_blocks  1.0 1.1 2.1 2.0   (wrong mapping: 2.1 1.0 1.1 2.0)
            # "input_blocks.4.1.transformer_blocks.0.attn1":
            #   "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor"
            for (a1, b1, a2, b2) in [(4, 1, 1, 0), (5, 1, 1, 1), (7, 1, 2, 1), (8, 1, 2, 0)]:
                for block_id in range(0, 10):
                    for attn in [1, 2]:
                        old_key = f"input_blocks.{a1}.{b1}.transformer_blocks.{block_id}.attn{attn}"
                        new_key = f"down_blocks.{a2}.attentions.{b2}.transformer_blocks.{block_id}.attn{attn}.processor"
                        replacement[old_key] = new_key
            # middle_block
            # "middle_block.1.transformer_blocks.0.attn1"
            # -> "mid_block.attentions.0.transformer_blocks.0.attn1.processor"
            for block_id in range(0, 10):
                for attn in [1, 2]:
                    old_key = f"middle_block.1.transformer_blocks.{block_id}.attn{attn}"
                    new_key = f"mid_block.attentions.0.transformer_blocks.{block_id}.attn{attn}.processor"
                    replacement[old_key] = new_key

            # output_blocks 0.1 1.1 2.1 3.1 4.1 5.1
            # up_blocks     0.0 0.1 0.2 1.0 1.1 1.2
            # "output_blocks.0.1.transformer_blocks.0.attn1":
            # -> "up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
            for (a1, b1, a2, b2) in [(0, 1, 0, 0), (1, 1, 0, 1), (2, 1, 0, 2), (3, 1, 1, 0), (4, 1, 1, 1), (5, 1, 1, 2)]:
                for block_id in range(0, 10):
                    for attn in [1, 2]:
                        old_key = f"output_blocks.{a1}.{b1}.transformer_blocks.{block_id}.attn{attn}"
                        new_key = f"up_blocks.{a2}.attentions.{b2}.transformer_blocks.{block_id}.attn{attn}.processor"
                        replacement[old_key] = new_key

            # Apply name replacement on all layers
            new_layer_lora_model = {}
            for layer_name, layer in layer_lora_model.items():
                new_layer_name = layer_name
                for old_substring, new_substring in replacement.items():
                    new_layer_name = new_layer_name.replace(old_substring, new_substring)
                new_layer_lora_model[new_layer_name] = layer

            # Save checkpoint with new key names
            logger.info(f"Write {new_layer_xl_transparent_attn_model_path}")
            safetensors.torch.save_file(new_layer_lora_model, new_layer_xl_transparent_attn_model_path)

            # Release layer_lora_model
            del layer_lora_model

        return new_layer_xl_transparent_attn_model_path

    def patch_with_layer_xl_transparent_attn(self) -> None:

        layer_xl_transparent_attn_model_path = self.create_layer_xl_transparent_attn_for_diffusers()

        # Load new checkpoint with correct key names
        logger.info(f"Load lora {layer_xl_transparent_attn_model_path}")
        self.load_lora_weights(layer_xl_transparent_attn_model_path)

    def load_transparent_vae_decoder(self) -> TransparentVAEDecoder:
        # Load TransparentVAEDecoder
        vae_transparent_decoder_model_path = load_file_from_url(
            url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
            model_dir=self.layer_model_cache_directory,
            file_name="vae_transparent_decoder.safetensors",
        )
        return TransparentVAEDecoder(safetensors.torch.load_file(vae_transparent_decoder_model_path))

    @staticmethod
    def load_scheduler() -> DPMSolverMultistepScheduler:
        # Scheduler parameters to match SD Forge implementation of DPM++ 2M SDE Karras
        scheduler_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
        }
        return DPMSolverMultistepScheduler(
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
            **scheduler_kwargs
        )

    @torch.no_grad()
    def _decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        This method is a duplicate of the code inside StableDiffusionXLPipeline.__call__(), between lines 1263-1293,
        with diffusers==0.28.1

        :param latents:
        :return:
        """
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Start of extra arguments
        return_intermediary_image: bool = False,
        use_augmentation_in_vae_transparent_decoder: bool = False,
        # End of extra arguments
        **kwargs,
    ):
        """

        See documentation in StableDiffusionXLPipeline.__call__().

        """
        # Input check
        if return_intermediary_image and output_type != "pil":
            raise ValueError(f"`return_intermediary_image=True` is only valid if `output_type=\"pil\"`")

        # Inference through StableDiffusionXLPipeline and return the latent (no VAE decoding)
        latents = super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type="latent",
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            kwargs=kwargs,
        ).images

        images_intermediary = None
        if output_type == "latent":
            image = latents
        else:
            # Decode latent into output RGB image with SDXL VAE decoder
            image = self._decode_latent(latents)

            # Denormalize image
            image = (image + 1.0) / 2.0

            # (Optional) Post process intermediary image
            if return_intermediary_image:
                # Restrict values in [0, 1] range
                images_intermediary = torch.clamp(image, min=0.0, max=1.0)
                images_intermediary = self.image_processor.postprocess(
                    images_intermediary,
                    output_type="pil",
                    do_denormalize=[False] * num_images_per_prompt
                )

            # Apply scaling factor to latent
            # -> Not sure about that, it depends on the range used during training, but I can't find information on this
            #    Running a few inferences with and without showed no difference
            latents = latents / self.vae.config.scaling_factor

            # Ensure image and latent have the same float depth as the Transparent VAE decoder
            vae_transparent_decoder_dtype = next(iter(self.vae_transparent_decoder.model.conv_in.parameters())).dtype
            image = image.to(dtype=vae_transparent_decoder_dtype)
            latents = latents.to(dtype=vae_transparent_decoder_dtype)

            # Infer through transparency VAE with latent and decoded RGB image
            if use_augmentation_in_vae_transparent_decoder:
                y = self.vae_transparent_decoder.estimate_augmented(image, latents)
            else:
                y = self.vae_transparent_decoder.estimate_single_pass(image, latents)

            # Extract alpha (1st channel) and foreground (2nd to 4th channels)
            alpha = y[:, :1, ...]
            fg = y[:, 1:, ...]
            # Reorder as RGBA
            image = torch.cat([fg, alpha], dim=1)
            # Restrict values in [0, 1] range
            image = torch.clamp(image, min=0.0, max=1.0)

            # The rest of this method is an adaptation of StableDiffusionXLPipeline.__call__() from line 1254 to 1266

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(
                image,
                output_type=output_type,
                do_denormalize=[False] * num_images_per_prompt
            )

        # Offload all models
        self.maybe_free_model_hooks()

        # Output logic
        if return_intermediary_image and images_intermediary is not None:
            if not return_dict:
                return image, images_intermediary
            else:
                image.extend(images_intermediary)
        elif not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
