import argparse, os, sys, glob
import shutil
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange

tqdm_auto_model = __import__("tqdm.auto", fromlist="")
sys.modules["tqdm"] = tqdm_auto_model
from einops import rearrange
from torchvision.utils import make_grid
import transformers
import gc

sys.path.append('./latent-diffusion')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.diffusionmodules.util import noise_like, make_ddim_sampling_parameters
import tensorflow as tf
from dotmap import DotMap
from math import pi

from resize_right import resize

import subprocess
from subprocess import Popen, PIPE, CalledProcessError

from dataclasses import dataclass
from functools import partial
import gc
import io
import math
import sys
import random
import tempfile
from piq import brisque
from itertools import product
import lpips
from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from numpy import nan
from threading import Thread
import time
import re
import base64
import warnings

import mmc
from mmc.registry import REGISTRY
import mmc.loaders  # force trigger model registrations
from mmc.mock.openai import MockOpenaiClip

model_path = "/root/.cache/majesty-diffusion"
outputs_path = "results"
device = None
opt = DotMap()

# Model
latent_diffusion_model = "finetuned"

# Change it to false to not use CLIP Guidance at all
use_cond_fn = True

# Custom cut schedules and super-resolution. Check out the guide on how to use it a https://multimodal.art/majestydiffusion
custom_schedule_setting = [
    [50, 1000, 8],
    "gfpgan:1.5",
    "scale:.9",
    "noise:.55",
    [5, 200, 5],
]

# Cut settings
clamp_index = [2.4, 2.1]  # linear variation of the index for clamping the gradient
cut_overview = [8] * 500 + [4] * 500
cut_innercut = [0] * 500 + [4] * 500
cut_ic_pow = 0.2
cut_icgray_p = [0.1] * 300 + [0] * 1000
cutn_batches = 1
cut_blur_n = [0] * 300 + [0] * 1000
cut_blur_kernel = 3
range_index = [0] * 200 + [5e4] * 400 + [0] * 1000
var_index = [2] * 300 + [0] * 700
var_range = 0.5
mean_index = [0] * 400 + [0] * 600
mean_range = 0.75
active_function = (
    "softsign"  # function to manipulate the gradient - help things to stablize
)
ths_method = "softsign"
tv_scales = [600] * 1 + [50] * 1 + [0] * 2

# If you uncomment next line you can schedule the CLIP guidance across the steps. Otherwise the clip_guidance_scale basic setting will be used
# clip_guidance_schedule = [10000]*300 + [500]*700

symmetric_loss_scale = 0  # Apply symmetric loss

# Latent Diffusion Advanced Settings
scale_div = 1  # Use when latent upscale to correct satuation problem
opt_mag_mul = 20  # Magnify grad before clamping
# PLMS Currently not working, working on a fix
opt_plms = False  # Experimental. It works but does not lookg good
opt_ddim_eta, opt_eta_end = [1.3, 1.1]  # linear variation of eta
opt_temperature = 0.98

# Grad advanced settings
grad_center = False
grad_scale = 0.25  # Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept

# Restraints the model from exploding despite larger clamp
score_modifier = True
threshold_percentile = 0.85
threshold = 1
score_corrector_setting = ["latent", ""]

# Init image advanced settings
init_rotate, mask_rotate = [False, False]
init_magnitude = 0.18215

# Noise settings
upscale_noise_temperature = 1
upscale_xT_temperature = 1

# More settings
RGB_min, RGB_max = [-0.95, 0.95]
padargs = {"mode": "constant", "value": -1}  # How to pad the image with cut_overview
flip_aug = False
cutout_debug = False

# Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
experimental_aesthetic_embeddings = True
# How much you want this to influence your result
experimental_aesthetic_embeddings_weight = 0.3
# 9 are good aesthetic embeddings, 0 are bad ones
experimental_aesthetic_embeddings_score = 8

# For fun dont change except if you really know what your are doing
grad_blur = False
compress_steps = 200
compress_factor = 0.1
punish_steps = 200
punish_factor = 0.5

# Amp up your prompt game with prompt engineering, check out this guide: https://matthewmcateer.me/blog/clip-prompt-engineering/
# Prompt for CLIP Guidance
clip_prompts = ["The portrait of a Majestic Princess, trending on artstation"]

# Prompt for Latent Diffusion
latent_prompts = ["The portrait of a Majestic Princess, trending on artstation"]

# Negative prompts for Latent Diffusion
latent_negatives = [""]

image_prompts = []

width = 256
height = 256
latent_diffusion_guidance_scale = 12
clip_guidance_scale = 16000
how_many_batches = 1
aesthetic_loss_scale = 400
augment_cuts = True
n_samples = 1

init_image = None
starting_timestep = 0.9
init_mask = None
init_scale = 1000
init_brightness = 0.0
init_noise = 0.57

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

# Globals
custom_settings = None
generate_video = False
model = {}
aes_scale = None
aug = None

clip_load_list, clip_guidance_index = [], []

aesthetic_model_336, aesthetic_model_224, aesthetic_model_16, aesthetic_model_32 = (
    {},
    {},
    {},
    {},
)
custom_schedules = []

progress = None
image_grid, writer, img_tensor, im = {}, {}, {}, {}
target_embeds, weights, zero_embed, init = {}, {}, {}, {}
make_cutouts = {}
scale_factor = 1
clamp_start_, clamp_max = None, None
clip_guidance_schedule = None
prompts = []
last_step_uspcale_list = []

has_purged = True

# Used to override download locations, allows rehosting models in a bucket for ephemeral servers to download
model_source = None


def download_models(ongo=False, erlich=False):
    # download models as needed
    models = [
        [
            "latent_diffusion_txt2img_f8_large.ckpt",
            "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt",
        ],
        [
            "txt2img-f8-large-jack000-finetuned-fp16.ckpt",
            "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-jack000-finetuned-fp16.ckpt",
        ],
        [
            "ava_vit_l_14_336_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_336_linear.pth",
        ],
        [
            "sa_0_4_vit_l_14_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_l_14_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_b_16_linear.pth",
            "http://batbot.tv/ai/models/v-diffusion/ava_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_16_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_32_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_32_linear.pth",
        ],
        [
            "openimages_512x_png_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/openimages_512x_png_embed224.npz",
        ],
        [
            "imagenet_512x_jpg_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/imagenet_512x_jpg_embed224.npz",
        ],
        [
            "GFPGANv1.3.pth",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        ],
    ]
    if ongo:
        models.append(
            [
                "ongo.pt",
                "https://huggingface.co/laion/ongo/resolve/main/ongo.pt",
            ]
        )
    if erlich:
        models.append(
            [
                "erlich.pt",
                "https://huggingface.co/laion/erlich/resolve/main/model/ema_0.9999_120000.pt",
            ]
        )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for item in models:
        model_file = f"{model_path}/{item[0]}"
        if not os.path.exists(model_file):
            if model_source:
                url = f"{model_source}/{item[0]}"
            else:
                url = item[1]
            print(f"Downloading {url}")
            subprocess.call(
                ["wget", "-nv", "-O", model_file, url, "--no-check-certificate"],
                shell=False,
            )
    if not os.path.exists("GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
        shutil.copyfile(
            f"{model_path}/GFPGANv1.3.pth",
            "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
        )


def load_model_from_config(
    config, ckpt, verbose=False, latent_diffusion_model="original"
):
    print(f"Loading model from {ckpt}")
    print(latent_diffusion_model)
    model = instantiate_from_config(config.model)
    if latent_diffusion_model != "finetuned":
        sd = torch.load(ckpt, map_location="cuda")["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)

    if latent_diffusion_model == "finetuned":
        sd = torch.load(
            f"{model_path}/txt2img-f8-large-jack000-finetuned-fp16.ckpt",
            map_location="cuda",
        )
        m, u = model.load_state_dict(sd, strict=False)
        # model.model = model.model.half().eval().to(device)

    if latent_diffusion_model.startswith("ongo"):
        del sd
        sd_finetuned = torch.load(f"{model_path}/ongo.pt")
        sd_finetuned["input_blocks.0.0.weight"] = sd_finetuned[
            "input_blocks.0.0.weight"
        ][:, 0:4, :, :]
        model.model.diffusion_model.load_state_dict(sd_finetuned, strict=False)
        del sd_finetuned
        torch.cuda.empty_cache()
        gc.collect()

    if latent_diffusion_model.startswith("erlich"):
        del sd
        sd_finetuned = torch.load(f"{model_path}/erlich.pt")
        sd_finetuned["input_blocks.0.0.weight"] = sd_finetuned[
            "input_blocks.0.0.weight"
        ][:, 0:4, :, :]
        model.model.diffusion_model.load_state_dict(sd_finetuned, strict=False)
        del sd_finetuned
        torch.cuda.empty_cache()
        gc.collect()

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.requires_grad_(False).half().eval().to("cuda")
    return model


def get_mmc_models(clip_load_list):
    mmc_models = []
    for model_key in clip_load_list:
        if not model_key:
            continue
        arch, pub, m_id = model_key[1:-1].split(" - ")
        mmc_models.append(
            {
                "architecture": arch,
                "publisher": pub,
                "id": m_id,
            }
        )
    return mmc_models


def set_custom_schedules():
    global custom_schedules
    custom_schedules = []
    for schedule_item in custom_schedule_setting:
        if isinstance(schedule_item, list):
            custom_schedules.append(np.arange(*schedule_item))
        else:
            custom_schedules.append(schedule_item)


def parse_prompt(prompt):
    if (
        prompt.startswith("http://")
        or prompt.startswith("https://")
        or prompt.startswith("E:")
        or prompt.startswith("C:")
        or prompt.startswith("D:")
    ):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        WholeCrop=0,
        WC_Allowance=10,
        WC_Grey_P=0.2,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
        cut_blur_n=0,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.WholeCrop = WholeCrop
        self.WC_Allowance = WC_Allowance
        self.WC_Grey_P = WC_Grey_P
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.cut_blur_n = cut_blur_n
        self.augs = T.Compose(
            [
                # T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    # scale=(0.9,0.95),
                    fill=-1,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.RandomPerspective(p=1, interpolation = T.InterpolationMode.BILINEAR, fill=-1,distortion_scale=0.2),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ]
        )

    def forward(self, input):
        gray = transforms.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [input.shape[0], 3, self.cut_size, self.cut_size]
        output_shape_2 = [input.shape[0], 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
            ),
            **padargs,
        )
        cutouts_list = []

        if self.Overview > 0:
            cutouts = []
            cutout = resize(pad_input, out_shape=output_shape, antialiasing=True)
            output_shape_all = list(output_shape)
            output_shape_all[0] = self.Overview * input.shape[0]
            pad_input = pad_input.repeat(input.shape[0], 1, 1, 1)
            cutout = resize(pad_input, out_shape=output_shape_all)
            if aug:
                cutout = self.augs(cutout)
            if self.cut_blur_n > 0:
                cutout[0 : self.cut_blur_n, :, :, :] = TF.gaussian_blur(
                    cutout[0 : self.cut_blur_n, :, :, :], cut_blur_kernel
                )
            cutouts_list.append(cutout)

        if self.InnerCrop > 0:
            cutouts = []
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].add(1).div(2).clamp(0, 1).squeeze(0)).save(
                    "content/diff/cutouts/cutout_InnerCrop.jpg", quality=99
                )
            cutouts_tensor = torch.cat(cutouts)
            cutouts = []
            cutouts_list.append(cutouts_tensor)
        cutouts = torch.cat(cutouts_list)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


# def range_loss(input, range_min, range_max):
#    return ((input - input.clamp(range_min,range_max)).abs()*10).pow(2).mean([1, 2, 3])
def range_loss(input, range_min, range_max):
    return ((input - input.clamp(range_min, range_max)).abs()).mean([1, 2, 3])


def symmetric_loss(x):
    w = x.shape[3]
    diff = (x - torch.flip(x, [3])).square().mean().sqrt() / (
        x.shape[2] * x.shape[3] / 1e4
    )
    return diff


def fetch(url_or_path):
    """Fetches a file from an HTTP or HTTPS url, or opens the local file."""
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub("^data:image/.+;base64,", "", base64_str)
    binary_data = base64.b64decode(base64_data)
    img_data = io.BytesIO(binary_data)
    img = Image.open(img_data)
    if image_path:
        img.save(image_path)
    return img


def centralized_grad(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


def cond_fn(x, t):
    global cur_step
    cur_step += 1
    t = 1000 - t
    t = t[0]
    x = x.detach()
    with torch.enable_grad():
        global clamp_start_, clamp_max
        x = x.detach()
        x = x.requires_grad_()
        x_in = model.decode_first_stage(x)
        display_handler(x_in, t, 1, False)
        n = x_in.shape[0]
        clip_guidance_scale = clip_guidance_index[t]
        make_cutouts = {}
        # rx_in_grad = torch.zeros_like(x_in)
        for i in clip_list:
            make_cutouts[i] = MakeCutouts(
                clip_size[i][0] if type(clip_size[i]) is tuple else clip_size[i],
                Overview=cut_overview[t],
                InnerCrop=cut_innercut[t],
                IC_Size_Pow=cut_ic_pow,
                IC_Grey_P=cut_icgray_p[t],
                cut_blur_n=cut_blur_n[t],
            )
            cutn = cut_overview[t] + cut_innercut[t]
        for j in range(cutn_batches):
            losses = 0
            for i in clip_list:
                clip_in = clip_normalize[i](
                    make_cutouts[i](x_in.add(1).div(2)).to("cuda")
                )
                image_embeds = (
                    clip_model[i]
                    .encode_image(clip_in)
                    .float()
                    .unsqueeze(0)
                    .expand([target_embeds[i].shape[0], -1, -1])
                )
                target_embeds_temp = target_embeds[i]
                if i == "ViT-B-32--openai" and experimental_aesthetic_embeddings:
                    aesthetic_embedding = torch.from_numpy(
                        np.load(
                            f"aesthetic-predictor/vit_b_32_embeddings/rating{experimental_aesthetic_embeddings_score}.npy"
                        )
                    ).to(device)
                    aesthetic_query = (
                        target_embeds_temp
                        + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    )
                    target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                        aesthetic_query
                    )
                if i == "ViT-L-14--openai" and experimental_aesthetic_embeddings:
                    aesthetic_embedding = torch.from_numpy(
                        np.load(
                            f"aesthetic-predictor/vit_l_14_embeddings/rating{experimental_aesthetic_embeddings_score}.npy"
                        )
                    ).to(device)
                    aesthetic_query = (
                        target_embeds_temp
                        + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    )
                    target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                        aesthetic_query
                    )
                target_embeds_temp = target_embeds_temp.unsqueeze(1).expand(
                    [-1, cutn * n, -1]
                )
                dists = spherical_dist_loss(image_embeds, target_embeds_temp)
                dists = dists.mean(1).mul(weights[i].squeeze()).mean()
                losses += dists * clip_guidance_scale
                if i == "ViT-L-14-336--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_336(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-L-14--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_224(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-B-16--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_16(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-B-32--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_32(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
            # x_in_grad += torch.autograd.grad(losses, x_in)[0] / cutn_batches / len(clip_list)
            # losses += dists
            # losses = losses / len(clip_list)
            # gc.collect()

        loss = losses
        # del losses
        if symmetric_loss_scale != 0:
            loss += symmetric_loss(x_in) * symmetric_loss_scale
        if init_image is not None and init_scale:
            lpips_loss = (lpips_model(x_in, init) * init_scale).squeeze().mean()
            # print(lpips_loss)
            loss += lpips_loss
        range_scale = range_index[t]
        range_losses = range_loss(x_in, RGB_min, RGB_max).sum() * range_scale
        loss += range_losses
        # loss_grad = torch.autograd.grad(loss, x_in, )[0]
        # x_in_grad += loss_grad
        # grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        loss.backward()
        grad = -x.grad
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0, neginf=0)
        if grad_center:
            grad = centralized_grad(grad, use_gc=True, gc_conv_only=False)
        mag = grad.square().mean().sqrt()
        if mag == 0 or torch.isnan(mag):
            print("ERROR")
            print(t)
            return grad
        if t >= 0:
            if active_function == "softsign":
                grad = F.softsign(grad * grad_scale / mag)
            if active_function == "tanh":
                grad = (grad / mag * grad_scale).tanh()
            if active_function == "clamp":
                grad = grad.clamp(-mag * grad_scale * 2, mag * grad_scale * 2)
        if grad.abs().max() > 0:
            grad = grad / grad.abs().max() * opt.mag_mul
            magnitude = grad.square().mean().sqrt()
        else:
            return grad
        clamp_max = clamp_index_variation[t]
        # print(magnitude, end = "\r")
        grad = grad * magnitude.clamp(max=clamp_max) / magnitude  # 0.2
        grad = grad.detach()
        grad = grad_fn(grad, t)
        x = x.detach()
        x = x.requires_grad_()
        var = x.var()
        var_scale = var_index[t]
        var_losses = (var.pow(2).clamp(min=var_range) - 1) * var_scale
        mean_scale = mean_index[t]
        mean_losses = (x.mean().abs() - mean_range).abs().clamp(min=0) * mean_scale
        tv_losses = (
            tv_loss(x).sum() * tv_scales[0]
            + tv_loss(F.interpolate(x, scale_factor=1 / 2)).sum() * tv_scales[1]
            + tv_loss(F.interpolate(x, scale_factor=1 / 4)).sum() * tv_scales[2]
            + tv_loss(F.interpolate(x, scale_factor=1 / 8)).sum() * tv_scales[3]
        )
        adjust_losses = tv_losses + var_losses + mean_losses
        adjust_losses.backward()
        grad -= x.grad
        # print(grad.abs().mean(), x.grad.abs().mean(), end = "\r")
        return grad


def null_fn(x_in):
    return torch.zeros_like(x_in)


def display_handler(x, i, cadance=5, decode=True):
    global progress, image_grid, writer, img_tensor, im, p
    img_tensor = x
    if i % cadance == 0:
        if decode:
            x = model.decode_first_stage(x)
        grid = make_grid(
            torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0),
            round(x.shape[0] ** 0.5 + 0.2),
        )
        grid = 255.0 * rearrange(grid, "c h w -> h w c").detach().cpu().numpy()
        image_grid = grid.copy(order="C")
        with io.BytesIO() as output:
            im = Image.fromarray(grid.astype(np.uint8))
            im.save(output, format="PNG")
            if progress:
                progress.value = output.getvalue()
            if generate_video:
                im.save(p.stdin, "PNG")


def grad_fn(x, t):
    if t <= 500 and grad_blur:
        x = TF.gaussian_blur(x, 2 * round(int(max(grad_blur - t / 150, 1))) - 1, 1.5)
    return x


def cond_clamp(image, t):
    t = 1000 - t[0]
    if t <= max(punish_steps, compress_steps):
        s = torch.quantile(
            rearrange(image, "b ... -> b (...)").abs(), threshold_percentile, dim=-1
        )
        s = s.view(-1, *((1,) * (image.ndim - 1)))
        ths = s.clamp(min=threshold)
        im_max = image.clamp(min=ths) - image.clamp(min=ths, max=ths)
        im_min = image.clamp(max=-ths, min=-ths) - image.clamp(max=-ths)
    if t <= punish_steps:
        image = (
            image.clamp(min=-ths, max=ths) + (im_max - im_min) * punish_factor
        )  # ((im_max-im_min)*punish_factor).tanh()/punish_factor
    if t <= compress_steps:
        image = image / (ths / threshold) ** compress_factor
        image += noise_like(image.shape, device, False) * (
            (ths / threshold) ** compress_factor - 1
        )
    return image


def make_schedule(t_start, t_end, step_size=1):
    schedule = []
    par_schedule = []
    t = t_start
    while t > t_end:
        schedule.append(t)
        t -= step_size
    schedule.append(t_end)
    return np.array(schedule)


def list_mul_to_array(list_mul):
    i = 0
    mul_count = 0
    mul_string = ""
    full_list = list_mul
    full_list_len = len(full_list)
    for item in full_list:
        if i == 0:
            last_item = item
        if item == last_item:
            mul_count += 1
        if item != last_item or full_list_len == i + 1:
            mul_string = mul_string + f" [{last_item}]*{mul_count} +"
            mul_count = 1
        last_item = item
        i += 1
    return mul_string[1:-2]


def generate_settings_file(add_prompts=False, add_dimensions=False):

    if add_prompts:
        prompts = f"""
    clip_prompts = {clip_prompts}
    latent_prompts = {latent_prompts}
    latent_negatives = {latent_negatives}
    image_prompts = []
    """
    else:
        prompts = ""

    if add_dimensions:
        dimensions = f"""width = {width}
    height = {height}
    """
    else:
        dimensions = ""
    settings = f"""
    #This settings file can be loaded back to Latent Majesty Diffusion. If you like your setting consider sharing it to the settings library at https://github.com/multimodalart/MajestyDiffusion
    [model]
    latent_diffusion_model = {latent_diffusion_model}
    
    [clip_list]
    perceptors = {clip_load_list}
    
    [basic_settings]
    #Perceptor things
    {prompts}
    {dimensions}
    latent_diffusion_guidance_scale = {latent_diffusion_guidance_scale}
    clip_guidance_scale = {clip_guidance_scale}
    aesthetic_loss_scale = {aesthetic_loss_scale}
    augment_cuts={augment_cuts}

    #Init image settings
    starting_timestep = {starting_timestep}
    init_scale = {init_scale} 
    init_brightness = {init_brightness}

    [advanced_settings]
    #Add CLIP Guidance and all the flavors or just run normal Latent Diffusion
    use_cond_fn = {use_cond_fn}

    #Custom schedules for cuts. Check out the schedules documentation here
    custom_schedule_setting = {custom_schedule_setting}

    #Cut settings
    clamp_index = {clamp_index}
    cut_overview = {list_mul_to_array(cut_overview)}
    cut_innercut = {list_mul_to_array(cut_innercut)}
    cut_blur_n = {list_mul_to_array(cut_blur_n)}
    cut_blur_kernel = {cut_blur_kernel}
    cut_ic_pow = {cut_ic_pow}
    cut_icgray_p = {list_mul_to_array(cut_icgray_p)}
    cutn_batches = {cutn_batches}
    range_index = {list_mul_to_array(range_index)}
    active_function = "{active_function}"
    ths_method= "{ths_method}"
    tv_scales = {list_mul_to_array(tv_scales)}    

    #If you uncomment this line you can schedule the CLIP guidance across the steps. Otherwise the clip_guidance_scale will be used
    clip_guidance_schedule = {list_mul_to_array(clip_guidance_index)}
    
    #Apply symmetric loss (force simmetry to your results)
    symmetric_loss_scale = {symmetric_loss_scale} 

    #Latent Diffusion Advanced Settings
    #Use when latent upscale to correct satuation problem
    scale_div = {scale_div}
    #Magnify grad before clamping by how many times
    opt_mag_mul = {opt_mag_mul}
    opt_ddim_eta = {opt_ddim_eta}
    opt_eta_end = {opt_eta_end}
    opt_temperature = {opt_temperature}

    #Grad advanced settings
    grad_center = {grad_center}
    #Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept
    grad_scale={grad_scale} 
    score_modifier = {score_modifier}
    threshold_percentile = {threshold_percentile}
    threshold = {threshold}
    var_index = {list_mul_to_array(var_index)}
    mean_index = {list_mul_to_array(mean_index)}
    mean_range = {mean_range}
    
    #Init image advanced settings
    init_rotate={init_rotate}
    mask_rotate={mask_rotate}
    init_magnitude = {init_magnitude}

    #More settings
    RGB_min = {RGB_min}
    RGB_max = {RGB_max}
    #How to pad the image with cut_overview
    padargs = {padargs} 
    flip_aug={flip_aug}
    
    #Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
    experimental_aesthetic_embeddings = {experimental_aesthetic_embeddings}
    #How much you want this to influence your result
    experimental_aesthetic_embeddings_weight = {experimental_aesthetic_embeddings_weight}
    #9 are good aesthetic embeddings, 0 are bad ones
    experimental_aesthetic_embeddings_score = {experimental_aesthetic_embeddings_score}

    # For fun dont change except if you really know what your are doing
    grad_blur = {grad_blur}
    compress_steps = {compress_steps}
    compress_factor = {compress_factor}
    punish_steps = {punish_steps}
    punish_factor = {punish_factor}
    """
    return settings


def load_clip_models(mmc_models):
    clip_model, clip_size, clip_tokenize, clip_normalize = {}, {}, {}, {}
    clip_list = []
    for item in mmc_models:
        print("Loaded ", item["id"])
        clip_list.append(item["id"])
        model_loaders = REGISTRY.find(**item)
        for model_loader in model_loaders:
            clip_model_loaded = model_loader.load()
            clip_model[item["id"]] = MockOpenaiClip(clip_model_loaded)
            clip_size[item["id"]] = clip_model[item["id"]].visual.input_resolution
            clip_tokenize[item["id"]] = clip_model[item["id"]].preprocess_text()
            clip_normalize[item["id"]] = normalize
    return clip_model, clip_size, clip_tokenize, clip_normalize, clip_list


def full_clip_load(clip_load_list):
    torch.cuda.empty_cache()
    gc.collect()
    try:
        del clip_model, clip_size, clip_tokenize, clip_normalize, clip_list
    except:
        pass
    mmc_models = get_mmc_models(clip_load_list)
    clip_model, clip_size, clip_tokenize, clip_normalize, clip_list = load_clip_models(
        mmc_models
    )
    return clip_model, clip_size, clip_tokenize, clip_normalize, clip_list


# Alstro's aesthetic model
def load_aesthetic_model():
    global aesthetic_model_336, aesthetic_model_224, aesthetic_model_16, aesthetic_model_32
    aesthetic_model_336 = torch.nn.Linear(768, 1).cuda()
    aesthetic_model_336.load_state_dict(
        torch.load(f"{model_path}/ava_vit_l_14_336_linear.pth")
    )

    aesthetic_model_224 = torch.nn.Linear(768, 1).cuda()
    aesthetic_model_224.load_state_dict(
        torch.load(f"{model_path}/ava_vit_l_14_linear.pth")
    )

    aesthetic_model_16 = torch.nn.Linear(512, 1).cuda()
    aesthetic_model_16.load_state_dict(
        torch.load(f"{model_path}/ava_vit_b_16_linear.pth")
    )

    aesthetic_model_32 = torch.nn.Linear(512, 1).cuda()
    aesthetic_model_32.load_state_dict(
        torch.load(f"{model_path}/sa_0_4_vit_b_32_linear.pth")
    )


def load_lpips_model():
    global lpips_model
    lpips_model = lpips.LPIPS(net="vgg").to(device)


def config_init_image():
    global custom_schedule_setting
    if (
        ((init_image is not None) and (init_image != "None") and (init_image != ""))
        and starting_timestep != 1
        and custom_schedule_setting[0][1] == 1000
    ):
        custom_schedule_setting[0] = [
            custom_schedule_setting[0][0],
            int(custom_schedule_setting[0][1] * starting_timestep),
            custom_schedule_setting[0][2],
        ]


def config_clip_guidance():
    global clip_guidance_index, clip_guidance_schedule, clip_guidance_scale
    if clip_guidance_schedule:
        clip_guidance_index = clip_guidance_schedule
    else:
        clip_guidance_index = [clip_guidance_scale] * 1000


def config_output_size():
    global opt
    opt.W = (width // 64) * 64
    opt.H = (height // 64) * 64
    if opt.W != width or opt.H != height:
        print(
            f"Changing output size to {opt.W}x{opt.H}. Dimensions must by multiples of 64."
        )


def config_options():
    global aes_scale, opt, aug, clamp_index_variation, score_corrector
    aes_scale = aesthetic_loss_scale
    opt.mag_mul = opt_mag_mul
    opt.ddim_eta = opt_ddim_eta
    opt.eta_end = opt_eta_end
    opt.temperature = opt_temperature
    opt.n_iter = 1
    opt.n_samples = n_samples
    opt.scale = latent_diffusion_guidance_scale
    opt.plms = opt_plms
    aug = augment_cuts
    # Checks if it's not a normal schedule (legacy purposes to keep old configs compatible)
    if len(clamp_index) == 2:
        clamp_index_variation = np.linspace(clamp_index[0], clamp_index[1], 1000)
    else:
        clamp_index_variation = clamp_index
    score_corrector = DotMap()
    score_corrector.modify_score = modify_score


def modify_score(e_t, e_t_uncond):
    if score_modifier is False:
        return e_t
    else:
        e_t_d = e_t - e_t_uncond
        s = torch.quantile(
            rearrange(e_t_d, "b ... -> b (...)").abs().float(),
            threshold_percentile,
            dim=-1,
        )

    s.clamp_(min=1.0)
    s = s.view(-1, *((1,) * (e_t_d.ndim - 1)))
    if ths_method == "softsign":
        e_t_d = F.softsign(e_t_d) / s
    elif ths_method == "clamp":
        e_t_d = e_t_d.clamp(-s, s) / s * 1.3  # 1.2
    e_t = e_t_uncond + e_t_d
    return e_t


def use_args(args: argparse.Namespace):
    global_var_scope = globals()
    warnings.filterwarnings("ignore")
    for k, v in vars(args).items():
        global_var_scope[k] = v


def load_custom_settings():
    global_var_scope = globals()
    global clip_load_list
    warnings.filterwarnings("ignore")
    if (
        custom_settings is not None
        and custom_settings != ""
        and custom_settings != "path/to/settings.cfg"
    ):
        print("Loaded ", custom_settings)
        try:
            from configparser import ConfigParser
        except ImportError:
            from ConfigParser import ConfigParser
        import configparser

        config = ConfigParser()
        config.read(custom_settings)
        # custom_settings_stream = fetch(custom_settings)
        # Load CLIP models from config
        if config.has_section("clip_list"):
            clip_incoming_list = config.items("clip_list")
            clip_incoming_models = clip_incoming_list[0]
            incoming_perceptors = eval(clip_incoming_models[1])
            if (len(incoming_perceptors) != len(clip_load_list)) or not all(
                elem in incoming_perceptors for elem in clip_load_list
            ):
                clip_load_list = incoming_perceptors
                has_purged = True

        # Load settings from config and replace variables
        if config.has_section("basic_settings"):
            basic_settings = config.items("basic_settings")
            for basic_setting in basic_settings:
                global_var_scope[basic_setting[0]] = eval(basic_setting[1])

        if config.has_section("advanced_settings"):
            advanced_settings = config.items("advanced_settings")
            for advanced_setting in advanced_settings:
                global_var_scope[advanced_setting[0]] = eval(advanced_setting[1])


def dynamic_thresholding(pred_x0, t):
    return pred_x0


def load_clip_globals(force=False):
    global has_purged
    if has_purged or force:
        global clip_model, clip_size, clip_tokenize, clip_normalize, clip_list
        (
            clip_model,
            clip_size,
            clip_tokenize,
            clip_normalize,
            clip_list,
        ) = full_clip_load(clip_load_list)
        has_purged = False

def do_run():
    load_clip_globals()        
    global opt, model, p, base_count, make_cutouts, progress, target_embeds, weights, zero_embed, init, scale_factor, cur_step, uc, c
    if generate_video:
        fps = 24
        p = Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vcodec",
                "libx264",
                "-r",
                str(fps),
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "17",
                "-preset",
                "veryslow",
                "video.mp4",
            ],
            stdin=PIPE,
        )
    #  with torch.cuda.amp.autocast():
    cur_step = 0
    scale_factor = 1
    make_cutouts = {}
    for i in clip_list:
        make_cutouts[i] = MakeCutouts(clip_size[i], Overview=1)
    for i in clip_list:
        target_embeds[i] = []
        weights[i] = []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        for i in clip_list:
            if "cloob" not in i:
                with torch.cuda.amp.autocast():
                    embeds = clip_model[i].encode_text(clip_tokenize[i](txt).to(device))
                    target_embeds[i].append(embeds)
                    weights[i].append(weight)
            else:
                embeds = clip_model[i].encode_text(clip_tokenize[i](txt).to(device))
                target_embeds[i].append(embeds)
                weights[i].append(weight)

    for prompt in image_prompts:
        if prompt.startswith("data:"):
            img = base64_to_image(prompt).convert("RGB")
            weight = 1
        else:
            print(f"processing{prompt}", end="\r")
            path, weight = parse_prompt(prompt)
            img = Image.open(fetch(path)).convert("RGB")
        img = TF.resize(
            img, min(opt.W, opt.H, *img.size), transforms.InterpolationMode.LANCZOS
        )
        for i in clip_list:
            if "cloob" not in i:
                with torch.cuda.amp.autocast():
                    batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
                    embed = clip_model[i].encode_image(clip_normalize[i](batch))
                    target_embeds[i].append(embed)
                    weights[i].extend([weight])
            else:
                batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
                embed = clip_model[i].encode_image(clip_normalize[i](batch))
                target_embeds[i].append(embed)
                weights[i].extend([weight])
    #    if anti_jpg != 0:
    #        target_embeds["ViT-B-32--openai"].append(
    #            torch.tensor(
    #                [
    #                    np.load(f"{model_path}/openimages_512x_png_embed224.npz")["arr_0"]
    #                    - np.load(f"{model_path}/imagenet_512x_jpg_embed224.npz")["arr_0"]
    #                ],
    #                device=device,
    #            )
    #        )
    #        weights["ViT-B-32--openai"].append(anti_jpg)

    for i in clip_list:
        target_embeds[i] = torch.cat(target_embeds[i])
        weights[i] = torch.tensor([weights[i]], device=device)
    shape = [4, opt.H // 8, opt.W // 8]
    init = None
    mask = None
    transform = T.GaussianBlur(kernel_size=3, sigma=0.4)
    if init_image is not None:
        if init_image.startswith("data:"):
            img = base64_to_image(init_image).convert("RGB")
        else:
            img = Image.open(fetch(init_image)).convert("RGB")
        init = TF.to_tensor(img).to(device).unsqueeze(0)
        if init_rotate:
            init = torch.rot90(init, 1, [3, 2])
        x0_original = torch.tensor(init)
        init = resize(init, out_shape=[opt.n_samples, 3, opt.H, opt.W])
        init = init.mul(2).sub(1).half()
        init_encoded = (
            model.first_stage_model.encode(init).sample() * init_magnitude
            + init_brightness
        )
        # init_encoded = init_encoded + noise_like(init_encoded.shape, device, False).mul(
        #    init_noise
        # )
        upscaled_flag = True
    else:
        init = None
        init_encoded = None
        upscaled_flag = False
    if init_mask is not None:
        mask = Image.open(fetch(init_mask)).convert("RGB")
        mask = TF.to_tensor(mask).to(device).unsqueeze(0)
        if mask_rotate:
            mask = torch.rot90(mask, 1, [3, 2])
        mask = F.interpolate(mask, [opt.H // 8, opt.W // 8]).mean(1)
        mask = transform(mask)
        print(mask)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.abspath(opt.outdir)

    prompt = opt.prompt
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples = list()
    last_step_upscale = False
    eta1 = opt.ddim_eta
    eta2 = opt.eta_end
    with torch.enable_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * opt.uc).cuda()

                for n in range(opt.n_iter):
                    torch.cuda.empty_cache()
                    gc.collect()
                    c = model.get_learned_conditioning(opt.n_samples * prompt).cuda()
                    if init_encoded is None:
                        x_T = torch.randn([opt.n_samples, *shape], device=device)
                        upscaled_flag = False
                        x0 = None
                    else:
                        x_T = init_encoded
                        x = torch.tensor(x_T)
                        upscaled_flag = True
                    last_step_uspcale_list = []
                    diffusion_stages = 0
                    for custom_schedule in custom_schedules:
                        if type(custom_schedule) != type(""):
                            diffusion_stages += 1
                            torch.cuda.empty_cache()
                            gc.collect()
                            last_step_upscale = False
                            samples_ddim, _ = sampler.sample(
                                S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                custom_schedule=custom_schedule,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=eta1
                                if diffusion_stages == 1 or last_step_upscale
                                else eta2,
                                eta_end=eta2,
                                img_callback=None if use_cond_fn else display_handler,
                                cond_fn=cond_fn if use_cond_fn else None,
                                temperature=opt.temperature,
                                x_adjust_fn=cond_clamp,
                                x_T=x_T,
                                x0=x_T,
                                mask=mask,
                                score_corrector=score_corrector,
                                corrector_kwargs=score_corrector_setting,
                                x0_adjust_fn=dynamic_thresholding,
                                clip_embed=target_embeds["ViT-L-14--openai"].mean(
                                    0, keepdim=True
                                )
                                if "ViT-L-14--openai" in clip_list
                                else None,
                            )
                            x_T = samples_ddim.clamp(-6, 6)
                            x_T = samples_ddim
                            last_step_upscale = False
                        else:
                            torch.cuda.empty_cache()
                            gc.collect()
                            method, scale_factor = custom_schedule.split(":")
                            if method == "RGB":
                                scale_factor = float(scale_factor)
                                temp_file_name = (
                                    "temp_" + f"{str(round(time.time()))}.png"
                                )
                                temp_file = os.path.join(sample_path, temp_file_name)
                                im.save(temp_file, format="PNG")
                                init = Image.open(fetch(temp_file)).convert("RGB")
                                init = TF.to_tensor(init).to(device).unsqueeze(0)
                                opt.H, opt.W = (
                                    opt.H * scale_factor,
                                    opt.W * scale_factor,
                                )
                                init = resize(
                                    init,
                                    out_shape=[opt.n_samples, 3, opt.H, opt.W],
                                    antialiasing=True,
                                )
                                init = init.mul(2).sub(1).half()
                                x_T = (
                                    model.first_stage_model.encode(init).sample()
                                    * init_magnitude
                                )
                                upscaled_flag = True
                                last_step_upscale = True
                                # x_T += noise_like(x_T.shape,device,False)*init_noise
                                # x_T = x_T.clamp(-6,6)
                            if method == "gfpgan":
                                scale_factor = float(scale_factor)
                                last_step_upscale = True
                                temp_file_name = (
                                    "temp_" + f"{str(round(time.time()))}.png"
                                )
                                temp_dir_name = tempfile.mkdtemp('gfpgan')
                                temp_file = os.path.join(temp_dir_name, temp_file_name)
                                im.save(temp_file, format="PNG")
                                GFP_factor = 2 if scale_factor > 1 else 1
                                GFP_ver = 1.3  # if GFP_factor == 1 else 1.2

                                torch.cuda.empty_cache()
                                gc.collect()
                                cmd = f"python inference_gfpgan.py -i {temp_file} -o results -v {GFP_ver} -s {GFP_factor}"
                                print(cmd + "\n")
                                try:
                                    subprocess.call(
                                        cmd,
                                        cwd="./GFPGAN",
                                        shell=True,
                                    )
                                except subprocess.CalledProcessError as e:
                                    print(f"GFPGAN execute failed {e.output}\n")

                                face_corrected = Image.open(
                                    fetch(
                                        f"GFPGAN/results/restored_imgs/{temp_file_name}"
                                    )
                                )
                                with io.BytesIO() as output:
                                    face_corrected.save(output, format="PNG")
                                    if progress:
                                        progress.value = output.getvalue()
                                init = Image.open(
                                    fetch(
                                        f"GFPGAN/results/restored_imgs/{temp_file_name}"
                                    )
                                ).convert("RGB")
                                init = TF.to_tensor(init).to(device).unsqueeze(0)
                                opt.H, opt.W = (
                                    opt.H * scale_factor,
                                    opt.W * scale_factor,
                                )
                                init = resize(
                                    init,
                                    out_shape=[opt.n_samples, 3, opt.H, opt.W],
                                    antialiasing=True,
                                )
                                init = init.mul(2).sub(1).half()
                                x_T = (
                                    model.first_stage_model.encode(init).sample()
                                    * init_magnitude
                                )
                                upscaled_flag = True
                                # x_T += noise_like(x_T.shape,device,False)*init_noise
                                # x_T = x_T.clamp(-6,6)
                                if method == "scale":
                                    scale_factor = float(scale_factor)
                                    x_T = x_T * scale_factor
                                if method == "noise":
                                    scale_factor = float(scale_factor)
                                    x_T += (
                                        noise_like(x_T.shape, device, False)
                                        * scale_factor
                                    )
                                if method == "purge":
                                    has_purged = True
                                    for i in scale_factor.split(","):
                                        if i in clip_load_list:
                                            arch, pub, m_id = i[1:-1].split(" - ")
                                            print("Purge ", i)
                                            del clip_list[clip_list.index(m_id)]
                                            del clip_model[m_id]
                                            del clip_size[m_id]
                                            del clip_tokenize[m_id]
                                            del clip_normalize[m_id]
                    # last_step_uspcale_list.append(last_step_upscale)
                    scale_factor = 1
                    current_time = str(round(time.time()))
                    if last_step_upscale and method == "gfpgan":
                        latest_upscale = Image.open(
                            fetch(f"GFPGAN/results/restored_imgs/{temp_file_name}")
                        ).convert("RGB")
                        latest_upscale.save(
                            os.path.join(outpath, f"{current_time}.png"), format="PNG"
                        )
                    else:
                        Image.fromarray(image_grid.astype(np.uint8)).save(
                            os.path.join(outpath, f"{current_time}.png"), format="PNG"
                        )
                    settings = generate_settings_file(
                        add_prompts=True, add_dimensions=False
                    )
                    text_file = open(f"{outpath}/{current_time}.cfg", "w")
                    text_file.write(settings)
                    text_file.close()
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    all_samples.append(x_samples_ddim)

    if len(all_samples) > 1:
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(
            os.path.join(outpath, f"grid_{str(round(time.time()))}.png")
        )

    if generate_video:
        p.stdin.close()
