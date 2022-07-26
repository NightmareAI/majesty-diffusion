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
import json
from tqdm.auto import tqdm, trange

sizes = [128, 192, 256, 320, 384]
model_path = "/root/.cache/majesty-diffusion"

default_perceptors = """
### Preloaded with image
[clip - mlfoundations - ViT-B-16--openai]
#[clip - mlfoundations - ViT-L-14--openai]
[clip - mlfoundations - ViT-B-32--laion2b_e16]
[clip - mlfoundations - ViT-L-14-336--openai]
### Download on demand
#[clip - mlfoundations - ViT-B-32--openai],
#[clip - mlfoundations - RN50x4--openai]
#[clip - mlfoundations - RN50x64--openai]
#[clip - mlfoundations - RN50x16--openai]
#[clip - mlfoundations - ViT-L-14--laion400m_e32]
#[clip - mlfoundations - ViT-B-16--laion400m_e32]
#[clip - mlfoundations - ViT-B-16-plus-240--laion400m_e32]
#[clip - sajjjadayobi - clipfa]
#[clip - navervision - kelip_ViT-B/32]
#[cloob - crowsonkb - cloob_laion_400m_vit_b_16_32_epochs]
"""

default_schedule = """[
[50, 1000, 8],
"gfpgan:1.5",
"scale:.9",
"noise:.55",
[5,300,4]            
]"""

default_settings = """
    [advanced_settings]
    #Add CLIP Guidance and all the flavors or just run normal Latent Diffusion
    use_cond_fn = True

    #Cut settings
    clamp_index = [2.4, 2.1]
    cut_overview = [8]*500 + [4]*500
    cut_innercut = [0]*500 + [4]*500
    cut_ic_pow = 0.2
    cut_icgray_p = [0.1]*300 + [0]*1000
    cutn_batches = 1
    cut_blur_n = [0] * 300 + [0] * 1000
    cut_blur_kernel = 3
    range_index = [0]*200+[5e4]*400+[0]*1000
    active_function = 'softsign'
    ths_method = 'softsign'
    tv_scales = [600] * 1 + [50] * 1 + [0] * 2
            
    #Apply symmetric loss (force simmetry to your results)
    symmetric_loss_scale = 0 

    #Latent Diffusion Advanced Settings
    #Use when latent upscale to correct satuation problem
    scale_div = 1
    #Magnify grad before clamping by how many times
    opt_mag_mul = 20
    opt_plms = False
    opt_ddim_eta = 1.3
    opt_eta_end = 1.1
    opt_temperature = 0.98

    #Grad advanced settings
    grad_center = False
    #Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept
    grad_scale=0.25
    score_modifier = True
    threshold_percentile = 0.85
    threshold = 1
    var_index = [2]*300+[0]*700
    var_range = 0.5
    mean_index = [0]*400+[0]*600
    mean_range = 0.75

    #Init image advanced settings
    init_rotate=False
    mask_rotate=False
    init_magnitude = 0.15

    #More settings
    RGB_min = -0.95
    RGB_max = 0.95
    #How to pad the image with cut_overview
    padargs = {'mode': 'constant', 'value': -1} 
    flip_aug=False
    
    #Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
    experimental_aesthetic_embeddings = True
    #How much you want this to influence your result
    experimental_aesthetic_embeddings_weight = 0.3
    #9 are good aesthetic embeddings, 0 are bad ones
    experimental_aesthetic_embeddings_score = 8

    # For fun dont change except if you really know what your are doing
    grad_blur = False
    compress_steps = 200
    compress_factor = 0.1
    punish_steps = 200
    punish_factor = 0.5
"""

class Predictor(BasePredictor):
    def load(self):
        config = OmegaConf.load("./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")    
        majesty.latent_diffusion_model = self.current_latent_diffusion_model        
        models.download_models(ongo=majesty.latent_diffusion_model == "ongo", erlich=majesty.latent_diffusion_model== "erlich")

        model = majesty.load_model_from_config(
            config,
            f"{majesty.model_path}/latent_diffusion_txt2img_f8_large.ckpt",
            False,
            self.current_latent_diffusion_model,
        )
        majesty.model = model.half().eval().to(majesty.device)
        majesty.load_lpips_model()
        majesty.load_aesthetic_model()
        torch.cuda.empty_cache()
        gc.collect()
        majesty.clip_load_list = self.current_clip_load_list
        majesty.load_clip_globals(True)
        

    def setup(self):
        print("Ensuring models are loaded..")
        self.current_latent_diffusion_model = "finetuned"
        self.current_clip_load_list = [line for line in default_perceptors.splitlines() if not line.strip().startswith("#")]
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
        clip_prompts: str = Input(
            description="Prompts for CLIP guidance, multiple prompts allowed, one per line. Supports weights.",
            default="The portrait of a Majestic Princess, trending on artstation\n",            
        ),
        latent_prompt: str = Input(
            description="Prompt for latent diffusion, single prompt, no weights.",
            default="The portrait of a Majestic Princess, trending on artstation",
            max_length=230,
        ),                
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
        clip_perceptors: str = Input(description="List of CLIP perceptor models to load, one per line. More models will consume more memory. Uses https://github.com/dmarx/Multi-Modal-Comparators.", default=default_perceptors),
        clip_scale: int = Input(description="CLIP guidance scale", default=16000),
        latent_scale: int = Input(
            description="The `latent_diffusion_guidance_scale` will determine how much the `latent_prompts` affect the image. Lower help with text interpretation, higher help with composition. Try values between 0-15. If you see too much text, lower it",
            default=12,
        ),        
        aesthetic_loss_scale: int = Input(description="Aesthetic loss scale", default=400),        
        starting_timestep: float = Input(description="Starting timestep", default=0.9),  
        model: str = Input(
            description="Latent diffusion model (ongo and erlich may need to download, taking extra time)",
            default="finetuned",
            choices=["original", "finetuned", "ongo", "erlich"],
        ),
        custom_schedule: str = Input(description="Custom schedules, JSON format. See the Majestic Guide for documentation.", default=default_schedule),
        latent_negatives: str = Input(description="Negative prompts for Latent Diffusion", default=""),      
        output_steps: int = Input(description="Steps between outputs, 0 to disable progressive output. Minor speed impact.", default=10, choices=[0, 5, 10, 20]),
        advanced_settings: str = Input(description="Advanced settings (can override values above)", default=default_settings, ),
    ) -> typing.Iterator[Path]:
        """Run a single prediction on the model"""

        try:
            clip_list = [line.strip() for line in clip_perceptors.splitlines() if line.strip().startswith("[") and line.strip().endswith("]")]
        except:            
            raise ValueError("Failed to parse clip_list!")
        if model != self.current_latent_diffusion_model or clip_list != self.current_clip_load_list:
            self.current_latent_diffusion_model = model
            self.current_clip_load_list = clip_list
            self.load()      

        try:
            schedule = json.loads(custom_schedule)
        except:
            raise ValueError("Failed to parse custom_schedule")

        majesty.clip_prompts = clip_prompts.splitlines()
        majesty.latent_prompts = [latent_prompt]
        majesty.latent_negatives = [latent_negatives]
        majesty.clip_guidance_scale = clip_scale
        majesty.latent_diffusion_guidance_scale = latent_scale
        majesty.aesthetic_loss_scale = aesthetic_loss_scale
        majesty.height = height
        majesty.width = width
        majesty.custom_schedule_setting = schedule
        majesty.starting_timestep = starting_timestep

        if init_image:
            majesty.init_image = str(init_image)
            majesty.init_scale = init_scale
            majesty.init_brightness = init_brightness
            if init_mask:
                majesty.init_mask = str(init_mask)

        outdir = tempfile.mkdtemp("majesty")
        majesty.opt.outdir = outdir

        if advanced_settings:
            settings_file = f'{outdir}/settings.cfg'
            with open(settings_file, 'w') as f:
                f.write(advanced_settings)
            majesty.custom_settings = settings_file

        majesty.load_custom_settings()
        majesty.config_init_image()
        majesty.prompts = majesty.clip_prompts
        majesty.opt.prompt = majesty.latent_prompts
        majesty.opt.uc = majesty.latent_negatives
        majesty.set_custom_schedules()
        majesty.config_clip_guidance()                        
                                
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