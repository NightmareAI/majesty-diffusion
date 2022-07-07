# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path, File
import tempfile
import glob
import models
import torch
import majesty
from omegaconf import OmegaConf
import os, shutil
import typing
import gc
import queue, threading
from tqdm.auto import tqdm, trange

sizes = [128, 192, 256, 320, 384]
model_path = "/root/.cache/majesty-diffusion"
current_latent_diffusion_model = "finetuned"
current_clip_load_list = [
    #    "[clip - mlfoundations - ViT-B-32--openai]",
    "[clip - mlfoundations - ViT-B-16--openai]",
    #    "[clip - mlfoundations - ViT-B-16--laion400m_e32]",
    "[clip - mlfoundations - ViT-L-14--openai]",
    #    "[clip - mlfoundations - ViT-L-14-336--openai]",
    "[clip - mlfoundations - ViT-B-32--laion2b_e16]",
]

class Predictor(BasePredictor):
    def load(self):
        config = OmegaConf.load("./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        majesty.latent_diffusion_model = current_latent_diffusion_model        
        models.download_models(ongo=majesty.latent_diffusion_model == "ongo", erlich=majesty.latent_diffusion_model== "erlich")

        model = majesty.load_model_from_config(
            config,
            f"{majesty.model_path}/latent_diffusion_txt2img_f8_large.ckpt",
            False,
            current_latent_diffusion_model,
        )
        majesty.model = model.half().eval().to(majesty.device)
        majesty.load_lpips_model()
        majesty.load_aesthetic_model()
        torch.cuda.empty_cache()
        gc.collect()
        majesty.clip_load_list = current_clip_load_list
        majesty.load_clip_globals(True)
        

    def setup(self):
        print("Ensuring models are loaded..")
        models.download_models(model_path=model_path)
        if not os.path.exists("/src/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
            shutil.copyfile(
                f"{model_path}/GFPGANv1.3.pth",
                "/src/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
            )
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        majesty.device = device
        self.load()

    def predict(
        self,
        clip_prompt: str = Input(
            description="Prompt for CLIP guidance",
            default="The portrait of a Majestic Princess, trending on artstation",
            max_length=230,
        ),
        latent_prompt: str = Input(
            description="Prompt for latent diffusion",
            default="The portrait of a Majestic Princess, trending on artstation",
            max_length=230,
        ),        
        output_steps: int = Input(description="Steps between outputs, 0 to disable progressive output. Minor speed impact.", default=10, choices=[0, 5, 10, 20]),
        model: str = Input(
            description="Latent diffusion model (ongo and erlich may need to download, taking extra time)",
            default="finetuned",
            choices=["original", "finetuned", "ongo", "erlich"],
        ),
        latent_negatives: str = Input(description="Negative prompts for Latent Diffusion", default=""),
        height: int = Input(
            description="Output height (output will be scaled up 1.5x with default settings)",
            default=256,
            choices=sizes,
        ),
        width: int = Input(
            description="Output width (output will be scaled up 1.5x with default settings)", default=256, choices=sizes
        ),
        init_image: Path = Input(description="Initial image", default=None),
        init_mask: Path = Input(
            description="A mask same width and height as the original image with the color black indicating where to inpaint",
            default=None,
        ),
        init_scale: int = Input(
            description="Controls how much the init image should influence the final result. Experiment with values around 1000",
            default=1000,
        ),
        init_brightness: float = Input(description="Init image brightness", default=0.0),
        latent_scale: int = Input(
            description="The `latent_diffusion_guidance_scale` will determine how much the `latent_prompts` affect the image. Lower help with text interpretation, higher help with composition. Try values between 0-15. If you see too much text, lower it",
            default=12,
        ),
        clip_scale: int = Input(description="CLIP guidance scale", default=16000),
        aesthetic_loss_scale: int = Input(description="Aesthetic loss scale", default=400),
        starting_timestep: float = Input(description="Starting timestep", default=0.9),        
        custom_settings: Path = Input(description="Advanced settings file (will override any settings from above if present)", default=None),
    ) -> typing.Iterator[Path]:
        """Run a single prediction on the model"""

        if model != current_latent_diffusion_model:  # or clip
            self.load()

        majesty.clip_prompts = [clip_prompt]
        majesty.latent_prompts = [latent_prompt]
        majesty.latent_negatives = [latent_negatives]
        majesty.clip_guidance_scale = clip_scale
        majesty.latent_diffusion_guidance_scale = latent_scale
        majesty.aesthetic_loss_scale = aesthetic_loss_scale
        majesty.height = height
        majesty.width = width
        majesty.starting_timestep = starting_timestep

        if init_image:
            majesty.init_image = init_image
            majesty.init_scale = init_scale
            majesty.init_brightness = init_brightness
            if init_mask:
                majesty.init_mask = init_mask

        if custom_settings:
            majesty.custom_settings = custom_settings

        majesty.load_custom_settings()
        majesty.config_init_image()
        majesty.prompts = majesty.clip_prompts
        majesty.opt.prompt = majesty.latent_prompts
        majesty.opt.uc = majesty.latent_negatives
        majesty.set_custom_schedules()
        majesty.config_clip_guidance()                        

            
        
        outdir = tempfile.mkdtemp("majesty")
        majesty.opt.outdir = outdir
        majesty.config_output_size()
        majesty.config_options()
        torch.cuda.empty_cache()
        gc.collect()

        num_batches = 1
        for n in trange(num_batches, desc="Sampling"):
            print(f"Sampling images {n+1}/{num_batches}")
            if (output_steps):
                output = queue.SimpleQueue()
                progress_path = outdir + "/progress"
                os.makedirs(progress_path, exist_ok=True)            
                majesty.progress_fn = lambda img: output.put(img)
                self.running = True
                t = threading.Thread(target=self.worker, daemon=True)
                ix = 0                
                t.start()
                while t.is_alive():
                    try:
                        image = output.get(block=True, timeout=5)
                        if (ix % output_steps == 0):
                            filename = f'{progress_path}/{ix}.png'                            
                            outfile = open(filename, 'wb')
                            outfile.write(image)
                            yield Path(filename)                    
                        ix += 1                
                    except: {}
            else:
                majesty.do_run()
            yield Path(glob.glob(outdir + "/*.png")[0])

        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
    
    def worker(self):
        majesty.do_run()