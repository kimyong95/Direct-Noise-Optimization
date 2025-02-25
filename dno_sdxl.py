import torch
import torch.nn as nn
import torchvision
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import argparse
import torch.utils.checkpoint as checkpoint
import os, shutil
from PIL import Image
import time
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from rewards import RFUNCTIONS
import numpy as np
import json
import wandb
import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()

# sampling algorithm
class SequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
   
        model_kwargs = {
            "sample": torch.stack([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t, t], device = self.device),
            "encoder_hidden_states": prompt_embeds
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)
        direction = direction[0]

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]

def sequential_sampling(pipeline, unet, sampler, prompt_embeds, added_cond_kwargs, noise_vectors): 


    sampler.initialize(noise_vectors)

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        # model_output = pipeline.unet(**model_kwargs, added_cond_kwargs=added_cond_kwargs)
        model_output = checkpoint.checkpoint(unet, model_kwargs["sample"], model_kwargs["timestep"], model_kwargs["encoder_hidden_states"], None, None, None, None, added_cond_kwargs)
        sampler.step(model_output) 

    return sampler.get_last_sample()

class BatchSequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
        batch = len(self._samples[0])

        uncond_embeds = torch.stack([prompt_embeds[0]] * batch)
        cond_embeds = torch.stack([prompt_embeds[1]] * batch)
   
        model_kwargs = {
            "sample": torch.concat([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t] * 2 * batch, device = self.device),
            "encoder_hidden_states": torch.concat([uncond_embeds, cond_embeds])
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)
    
        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        self._samples = [self.noise_vectors[-1]]

def batch_sequential_sampling(pipeline, unet, sampler, prompt_embeds, added_cond_kwargs, noise_vectors): 


    sampler.initialize(noise_vectors)
    batch = noise_vectors.shape[1]
    uncond_text_embeds = torch.stack([added_cond_kwargs["text_embeds"][0]] * batch)
    cond_text_embeds = torch.stack([added_cond_kwargs["text_embeds"][1]] * batch)
    uncond_time_ids = torch.stack([added_cond_kwargs["time_ids"][0]] * batch)
    cond_time_ids = torch.stack([added_cond_kwargs["time_ids"][1]] * batch)

    added_cond_kwargs_batch = {
        "text_embeds": torch.concat([uncond_text_embeds, cond_text_embeds], dim = 0),
        "time_ids": torch.concat([uncond_time_ids, cond_time_ids], dim = 0)
    }

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        model_output = unet(added_cond_kwargs=added_cond_kwargs_batch, **model_kwargs)
        sampler.step(model_output) 

    return sampler.get_last_sample()

def decode_latent(decoder, latent):
    img = checkpoint.checkpoint(decoder.decode, latent / decoder.config.scaling_factor,  use_reentrant=False).sample
    return img

def find_closest_factors(n):
    # Start from the square root of n and move downwards to find the closest factors
    for i in range(int(math.sqrt(n)), n):
        if n % i == 0:
            return i
    return n

def log_images(dir, images, stage, step):
    images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
    image_dir = str(dir).removesuffix("/files") + f"/images/{stage}"
    os.makedirs(image_dir, exist_ok=True)
    grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
    torchvision.utils.save_image(grid_image, f"{image_dir}/{step}.jpg", format='jpeg')

def compute_probability_regularization(noise_vectors, eta, opt_time, subsample, shuffled_times = 100):
    
    
    # squential subsampling
    if eta > 0:
        noise_vectors_flat = noise_vectors[:(opt_time + 1)].flatten()
    else:
        noise_vectors_flat = noise_vectors[-1].flatten()
        
    dim = noise_vectors_flat.shape[0]

    # use for computing the probability regularization
    subsample_dim = round(4 ** subsample)
    subsample_num = dim // subsample_dim
        
    noise_vectors_seq = noise_vectors_flat.view(subsample_num, subsample_dim)

    seq_mean = noise_vectors_seq.mean(dim = 0)
    noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
    seq_cov = noise_vectors_seq.T @ noise_vectors_seq
    seq_var = seq_cov.diag()
    
    # compute the probability of the noise
    seq_mean_M = torch.norm(seq_mean)
    seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
    
    seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
    seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
    seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
    seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
    seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
    seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
    seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)

    shuffled_mean_prob_list = []
    shuffled_cov_prob_list = [] 
    
    shuffled_mean_log_prob_list = []
    shuffled_cov_log_prob_list = [] 
    
    shuffled_mean_M_list = []
    shuffled_cov_M_list = []

    for _ in range(shuffled_times):
        noise_vectors_flat_shuffled = noise_vectors_flat[torch.randperm(dim)]   
        noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
        
        shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
        noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
        shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
        shuffled_var = shuffled_cov.diag()
        
        # compute the probability of the noise
        shuffled_mean_M = torch.norm(shuffled_mean)
        shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
        

        shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
        shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
        shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
        shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
        
        shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
        shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
        shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
        
        
        shuffled_mean_prob_list.append(shuffled_mean_prob.item())
        shuffled_cov_prob_list.append(shuffled_cov_prob.item())
        
        shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
        shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
        
        shuffled_mean_M_list.append(shuffled_mean_M.item())
        shuffled_cov_M_list.append(shuffled_cov_M.item())
        
    reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)
    
    return reg_loss

def main():
    def list_of_strings(arg):
        return arg.split(";")
    parser = argparse.ArgumentParser(description='Diffusion Optimization with Differentiable Objective')
    parser.add_argument('--name', type=str, default="dno", help='name of the experiment')
    parser.add_argument('--prompt', type=str, default="A robotic cat and a robotic dog playing with a football on the moon. The cat wearing a hat, and dog wearing a scarf.", help='prompt for the optimization')
    parser.add_argument('--reward_prompts', type=list_of_strings, default="They playing with football on the moon.;A robotic cat and a robotic dog playing with a football on the moon.;A robotic cat and a robotic dog playing with a football on the moon. The dog wearing a scarf.;A robotic cat and a robotic dog playing with a football on the moon. The cat wearing a hat, and dog wearing a scarf.", help='prompt for the optimization')
    parser.add_argument('--eta', type=float, default=1.0, help='eta for the DDIM algorithm, eta=0 is ODE-based sampling while eta>0 is SDE-based sampling')
    parser.add_argument('--device', type=str, default="cuda", help='device for optimization')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--opt_steps', type=int, default=500, help='number of optimization steps')
    parser.add_argument('--objective', type=str, default="gemini-binary", help='objective for optimization', choices = RFUNCTIONS.keys())
    parser.add_argument('--mu', type=float, default=0.01, help='control the precison of gradient approxmiation')
    parser.add_argument('--gamma', type=float, default=0., help='coefficient for the probability regularization')
    parser.add_argument('--subsample', type=int, default=1, help='subsample factor for the computing the probability regularization')
    parser.add_argument('--lr', type=float, default=0.001, help='stepsize for optimization')
    parser.add_argument('--output', type=str, default="output", help='output path')
    parser.add_argument('--sd_model', type=str, default="sdxl", help='model for the stable diffusion', choices = ["sdxl", "sdxl-lightning"])
    parser.add_argument('--grad_estimate_batchsize', type=int, default=8, help='batch size per device')
    parser.add_argument('--grad_estimate_total_num', type=int, default=32, help='number of samples for gradient estimation')
    parser.add_argument('--load_init_noise', type=str, default=None, help='load init noise')
    args = parser.parse_args()

    wandb_name = args.name

    run = wandb.init(
        project="guide-stable-diffusion",
        name=wandb_name,
        config=args,
    )

    if args.sd_model == "sdxl":
        num_sampling_steps = 30
        guidance_scale = 7.0
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device = args.device)
        pipeline.enable_vae_slicing()
    elif args.sd_model == "sdxl-lightning":
        num_sampling_steps = 8
        guidance_scale = 1.0
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device = args.device)
        pipeline.enable_vae_slicing()

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    # set the number of steps
    pipeline.scheduler.set_timesteps(num_sampling_steps)
    unet = pipeline.unet

    # load the loss function, which is negative of the reward fucntion
    loss_fn = RFUNCTIONS[args.objective](inference_dtype = torch.float32, device = args.device)

    torch.manual_seed(args.seed)

    if args.load_init_noise is not None:
        epsilon = torch.load(args.load_init_noise)
        noise_vectors = epsilon.flip(0).to(args.device)
    else:
        noise_vectors = torch.randn(num_sampling_steps + 1, 4, 128, 128, device = args.device)
    
    noise_vectors.requires_grad_(True)
    optimize_groups = [{"params":noise_vectors, "lr":args.lr}]
    optimizer = torch.optim.AdamW(optimize_groups)
    
    (prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,    
        ) = pipeline.encode_prompt(
                        prompt = args.prompt,
                        device = args.device
                    )
        
    
    # Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    add_time_ids = pipeline._get_add_time_ids(
            (1024, 1024),
            (0, 0),
            (1024, 1024),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    negative_add_time_ids = add_time_ids
    
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(args.device)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0).to(args.device)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(args.device)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # start optimization, opt fpr using fp16 mixed precision
    use_amp = True
    grad_scaler = GradScaler(enabled=use_amp, init_scale = 8192)
    amp_dtype = torch.float16
    
    use_mu = args.mu
    
    for step in range(args.opt_steps):
        
        optimizer.zero_grad()

        ############################# Calculate sample #############################
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            ddim_sampler = SequentialDDIM(timesteps = num_sampling_steps,
                                            scheduler = pipeline.scheduler, 
                                            eta = args.eta, 
                                            cfg_scale = guidance_scale, 
                                            device = args.device,
                                            opt_timesteps = num_sampling_steps)

            sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds,added_cond_kwargs = added_cond_kwargs, noise_vectors = noise_vectors)
            sample = decode_latent(pipeline.vae, sample.unsqueeze(0))[0]
            sample_image = pipeline.image_processor.postprocess(sample.unsqueeze(0).detach(), output_type="pil")[0]

        with torch.no_grad():
            reward = - loss_fn([sample_image], args.reward_prompts[-1]).item()
        ############################################################################

        ############################ Estimate gradient #############################
        with torch.no_grad():
            assert args.grad_estimate_total_num % args.grad_estimate_batchsize == 0
            grad_estimate_total_num = args.grad_estimate_total_num
            grad_estimate_batchsize = args.grad_estimate_batchsize
            num_batches = grad_estimate_total_num // grad_estimate_batchsize

            noise_vectors_flat = noise_vectors.detach().unsqueeze(1).to(dtype=torch.float16)
            cand_noise_vectors = noise_vectors_flat + use_mu * torch.randn(num_sampling_steps + 1, grad_estimate_total_num - 1 , 4, 128, 128, device = args.device, dtype = torch.float16)
            cand_noise_vectors = torch.concat([cand_noise_vectors, noise_vectors_flat], dim = 1)

            samples = []
            samples_image = []
            losses = []

            for i in range(num_batches):
                ddim_sampler = BatchSequentialDDIM(timesteps = num_sampling_steps,
                                                scheduler = pipeline.scheduler, 
                                                eta = args.eta, 
                                                cfg_scale = guidance_scale, 
                                                device = args.device,
                                                opt_timesteps = num_sampling_steps)
                
                i_start = i * grad_estimate_batchsize
                i_end = (i + 1) * grad_estimate_batchsize

                _samples = batch_sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, added_cond_kwargs = added_cond_kwargs, noise_vectors = cand_noise_vectors[:,i_start:i_end])
                _samples = decode_latent(pipeline.vae, _samples)
                _samples_image = pipeline.image_processor.postprocess(_samples, output_type="pil")
                _losses = []
                for reward_prompt in args.reward_prompts:
                    _loss = loss_fn(_samples_image, reward_prompt)
                    _losses.append(_loss)
                _losses = torch.stack(_losses, dim=0)

                samples.append(_samples)
                samples_image.extend(_samples_image)
                losses.append(_losses)

            samples = torch.cat(samples)
            losses = torch.cat(losses, dim = -1)
            losses_mean = losses.mean(dim = 0) # mean across prompts

            est_grad = torch.zeros_like(samples[-1])
            for i in range(grad_estimate_total_num):
                est_grad += (losses_mean[i] - losses_mean[-1]) * (samples[i] - samples[-1])
        
        est_grad = est_grad.unsqueeze(0)
        est_grad /= (torch.norm(est_grad) + 1e-3)

        ############################################################################

        ############################### Optimization ###############################

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):

            loss = torch.sum(est_grad * sample)
            
            if args.gamma > 0:
                reg_loss = compute_probability_regularization(noise_vectors, args.eta, num_sampling_steps, args.subsample)
                loss = loss + args.gamma * reg_loss

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            

        # sample_image.save(os.path.join(output_path, f"{step}_{reward}.png"))
        wandb_dir = run.dir
        log_images(wandb_dir, [sample_image], "validation", step)
        log_images(wandb_dir, samples_image, "train", step)

        save_noise_dir = str(wandb_dir).removesuffix("/files") + f"/checkpoints"
        os.makedirs(save_noise_dir, exist_ok=True)
        save_noise_path = os.path.join(save_noise_dir, f"noise_vectors_{step}.pt")
        torch.save(noise_vectors, save_noise_path)

        print(f"step : {step}, reward : {reward}")
        wandb.log({
            "epoch": step,
            "train/score_mean": losses_mean.mean(),
            "score_f0_mean": losses[0].mean(),
            "score_f1_mean": losses[1].mean(),
            "score_f2_mean": losses[2].mean(),
            "score_f3_mean": losses[3].mean(),
            "validation/score_mean": -reward,
        })
        ############################################################################

if __name__ == "__main__":
    main()