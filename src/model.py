import sys
import time

sys.path.extend(
    ["./src/vendor/taming-transformers", "./src/vendor/stable-diffusion", "./src/vendor/latent-diffusion"])

import huggingface_hub
import k_diffusion as K
import numpy as np
import torch
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from requests.exceptions import HTTPError
from torch import nn
from torchvision.transforms import functional as TF

from tools import fetch, resize_image_if_needed, save_image


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(
            1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(
            low_res, scale_factor=2, mode="nearest") * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


def make_upscaler_model(
    config_path, model_path, pooler_dim=768, train=False, device="cpu"
):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config["model"]["sigma_data"],
        embed_dim=config["model"]["mapping_cond_dim"] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_ema"])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def download_from_huggingface(repo, filename):
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename)
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(
                x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in
            )

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item])
                for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(
            x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"][indexer]
        attention_mask = 1 - tok_out["attention_mask"][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )


# Load models on GPU
def load_model_from_config(config, ckpt, cpu):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to(cpu).eval().requires_grad_(False)
    return model


class StableDiffusionUpscaler:
    def __init__(self, config, cpu, device):
        self.config = config
        self.tok_up = CLIPTokenizerTransform()
        self.text_encoder_up = CLIPEmbedder(device=device)
        self.cpu = cpu
        self.device = device
        self.load_model_on_gpu()
        self.upload_image_for_upscaling()

    def load_model_on_gpu(self):
        # sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
        vae_840k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt"
        )
        vae_560k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt"
        )

        # sd_model = load_model_from_config("stable-diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path)
        vae_model_840k = load_model_from_config(
            "src/vendor/latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_840k_model_path,
            self.cpu,
        )
        vae_model_560k = load_model_from_config(
            "src/vendor/latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_560k_model_path,
            self.cpu,
        )

        # sd_model = sd_model.to(device)
        self.vae_model_840k = vae_model_840k.to(self.device)
        self.vae_model_560k = vae_model_560k.to(self.device)
        url_config = (
            "https://models.rivershavewings.workers.dev/"
            "config_laion_text_cond_latent_upscaler_2.json"
        )
        url_model = (
            "https://models.rivershavewings.workers.dev/"
            "laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
        )
        model_up = make_upscaler_model(fetch(url_config), fetch(url_model))
        model_up = model_up.to(self.device)
        self.model_up = model_up

    def upload_image_for_upscaling(self):
        if self.config["input_image"] == "":
            self.config["input_image"] = "https://models.rivershavewings.workers.dev/assets/sd_2x_upscaler_demo.png"
        if self.config["input_image"].startswith("http://") or self.config["input_image"].startswith("https://"):
            input_image = Image.open(
                fetch(self.config["input_image"])
            ).convert("RGB")
        else:
            input_image = Image.open(
                self.config["input_image"]
            ).convert("RGB")
        self.input_image = resize_image_if_needed(input_image)

    @torch.no_grad()
    def condition_up(self, prompts):
        return self.text_encoder_up(self.tok_up(prompts))

    @torch.no_grad()
    def run(self):
        timestamp = int(time.time())
        if not self.config["seed"]:
            print("No seed was provided, using the current time.")
            self.config["seed"] = timestamp
        seed_everything(self.config["seed"])

        uc = self.condition_up(self.config["batch_size"] * [""])
        c = self.condition_up(
            self.config["batch_size"] * [self.config["prompt"]])

        if self.config["decoder"] == "finetuned_840k":
            vae = self.vae_model_840k
        elif self.config["decoder"] == "finetuned_560k":
            vae = self.vae_model_560k

        # image = Image.open(fetch(input_file)).convert('RGB')
        image = self.input_image
        image = TF.to_tensor(image).to(self.device) * 2 - 1
        low_res_latent = vae.encode(
            image.unsqueeze(0)).sample() * self.config["SD_Q"]
        # low_res_decoded = vae.decode(low_res_latent / config.SD_Q)

        [_, C, H, W] = low_res_latent.shape

        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(
            self.model_up, uc, cond_scale=self.config["guidance_scale"])
        low_res_sigma = torch.full(
            [self.config["batch_size"]
             ], self.config["noise_aug_level"], device=self.device
        )
        x_shape = [self.config["batch_size"], C, 2 * H, 2 * W]

        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min,
            # using one of the k diffusion samplers.
            sigmas = (
                torch.linspace(np.log(sigma_max), np.log(
                    sigma_min), self.config["steps"] + 1)
                .exp()
                .to(self.device)
            )
            if self.config["sampler"] == "k_euler":
                return K.sampling.sample_euler(
                    model_wrap, noise * sigma_max, sigmas, extra_args=extra_args
                )
            elif self.config["sampler"] == "k_euler_ancestral":
                return K.sampling.sample_euler_ancestral(
                    model_wrap,
                    noise * sigma_max,
                    sigmas,
                    extra_args=extra_args,
                    eta=self.config["eta"],
                )
            elif self.config["sampler"] == "k_dpm_2_ancestral":
                return K.sampling.sample_dpm_2_ancestral(
                    model_wrap,
                    noise * sigma_max,
                    sigmas,
                    extra_args=extra_args,
                    eta=self.config["eta"],
                )
            elif self.config["sampler"] == "k_dpm_fast":
                return K.sampling.sample_dpm_fast(
                    model_wrap,
                    noise * sigma_max,
                    sigma_min,
                    sigma_max,
                    self.config["steps"],
                    extra_args=extra_args,
                    eta=self.config["eta"],
                )
            elif self.config["sampler"] == "k_dpm_adaptive":
                sampler_opts = dict(
                    s_noise=1.0,
                    rtol=self.config["tol_scale"] * 0.05,
                    atol=self.config["tol_scale"] / 127.5,
                    pcoeff=0.2,
                    icoeff=0.4,
                    dcoeff=0,
                )
                return K.sampling.sample_dpm_adaptive(
                    model_wrap,
                    noise * sigma_max,
                    sigma_min,
                    sigma_max,
                    extra_args=extra_args,
                    eta=self.config["eta"],
                    **sampler_opts,
                )

        image_id = 0
        for _ in range((self.config["num_samples"] - 1) // self.config["batch_size"] + 1):
            if self.config["noise_aug_type"] == "gaussian":
                latent_noised = (
                    low_res_latent
                    + self.config["noise_aug_level"] *
                    torch.randn_like(low_res_latent)
                )
            elif self.config["noise_aug_type"] == "fake":
                latent_noised = (
                    low_res_latent *
                    (self.config["noise_aug_level"]**2 + 1) ** 0.5
                )
            extra_args = {
                "low_res": latent_noised,
                "low_res_sigma": low_res_sigma,
                "c": c,
            }
            noise = torch.randn(x_shape, device=self.device)
            up_latents = do_sample(noise, extra_args)

            # equivalent to sd_model.decode_first_stage(up_latents)
            pixels = vae.decode(up_latents / self.config["SD_Q"])
            pixels = pixels.add(1).div(2).clamp(0, 1)

            for j in range(pixels.shape[0]):
                img = TF.to_pil_image(pixels[j])
                save_image(
                    img,
                    timestamp=timestamp,
                    index=image_id,
                    prompt=self.config["prompt"],
                    seed=self.config["seed"],
                    save_location=self.config["save_location"],
                )
                image_id += 1
