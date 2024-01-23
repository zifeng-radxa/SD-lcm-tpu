import inspect

import cv2
from .npuengine import EngineOV
from .scheduler import diffusers_scheduler_config
import numpy as np
import torch
from transformers import CLIPTokenizer
from tqdm import tqdm
import threading
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    logging
)
import random
import time
import os
from multiprocessing import  Process,Pool
import multiprocessing
def seed_torch(seed=1029):
    seed=seed%4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

#seed_torch()
randn_tensor = lambda *x, **kwargs: torch.randn(*x, **kwargs)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def parallel_manager(pipe,model_path, device_id):
    model = EngineOV(model_path, device_id=device_id)
    while True:
        message = pipe.recv()
        if message['state'] == 1:
            data = message['data']
            # print("inner start ",time.time())
            res  = model(data)
            # print("inner end ",time.time())
            pipe.send({"from_sub":res})
        elif message['state'] == 2:
            break  
    del model

def parallel_maneger_prompt_embed(pipe,text_encoder_path,tokenizer_path, device_id):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    text_encoder = EngineOV(text_encoder_path, device_id=device_id)
    while True:
        message = pipe.recv()
        if message['state'] == 1:
            prompt = message['prompt']
            # =======================处理============================
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder({"tokens": np.array(text_input_ids)})
            prompt_embeds = prompt_embeds[0]
            # =======================结束============================
            pipe.send({"from_sub":prompt_embeds})
        elif message['state'] == 2:
            negative_prompt = message['negative_prompt']
            prompt = message['prompt']
            batch_size = message['batch_size']
            # =======================处理============================
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = tokenizer.model_max_length
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_text_input_ids = uncond_input.input_ids
            negative_prompt_embeds = text_encoder({"tokens": np.array(uncond_input_text_input_ids)})

            negative_prompt_embeds = negative_prompt_embeds[0]
            # =========================结束============================
            pipe.send({"from_sub":negative_prompt_embeds})

        elif message['state'] == 3:
            break
    del text_encoder

            
class SD15Parallel():
    def __init__(
        self,
        vae_encoder_path,
        vae_decoder_path,
        text_encoder_path,
        tokenizer_path,
        unet_path,
        device_id=0,
        parallel=False,
        is_tiny_vae = True
    ):
        # Tokenizer load
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.device_id = device_id
        self.parallel  = parallel
        
        # Unet load
        # self.unet = EngineOV(unet_path, device_id=self.device_id)
        if self.parallel == True:
            parent_pipe, child_pipe = multiprocessing.Pipe()
            self.child_pipe = child_pipe
            self.parent_pipe= parent_pipe
            process = multiprocessing.Process(target=parallel_manager, args=(child_pipe, unet_path, 1,))
            process.start()
            self.process = process
            # warmup
            self.parent_pipe.send({"state":10,"data":np.random.rand(50,60,70)})
            
            parent_pipe2, child_pipe2 = multiprocessing.Pipe()
            self.child_pipe2 = child_pipe2
            self.parent_pipe2= parent_pipe2
            process2 = multiprocessing.Process(target=parallel_manager, args=(child_pipe2, unet_path,0,))
            process2.start()
            self.process2 = process2
            self.parent_pipe2.send({"state":10,"data":np.random.rand(40,50,60,7)})
            
            # text部分多进程处理
            self.text_parent_pipe, self.text_child_pipe = multiprocessing.Pipe()
            text_process = multiprocessing.Process(target=parallel_maneger_prompt_embed, args=(self.text_child_pipe,text_encoder_path,tokenizer_path, 1,))
            text_process.start()
            self.text_process = text_process
            self.text_parent_pipe.send({"state":10,"data":np.random.rand(40,50,60,7)})

            
            self.text_parent_pipe2, self.text_child_pipe2 = multiprocessing.Pipe()
            text_process2 = multiprocessing.Process(target=parallel_maneger_prompt_embed, args=(self.text_child_pipe2,text_encoder_path,tokenizer_path, 0,))
            text_process2.start()
            self.text_process2 = text_process2
            self.text_parent_pipe2.send({"state":10,"data":np.random.rand(40,50,60,7)})


        # VAE load
        self.vae_decoder = EngineOV(vae_decoder_path, device_id=self.device_id)
        self.vae_encoder = EngineOV(vae_encoder_path, device_id=self.device_id)
        
        # Text_Encoder load
        self.text_encoder_path = text_encoder_path
        self.text_encoder = EngineOV(text_encoder_path, device_id=self.device_id)

        # Scheculer
        self.scheduler = EulerDiscreteScheduler(**(diffusers_scheduler_config['Euler D']))
        
        # other config
        self.config_force_zeros_for_empty_prompt = False
        self.prompt_embeds_dtype = torch.float16
        self.vae_config_scaling_factor = 0.18215
        if is_tiny_vae:
            self.vae_config_scaling_factor = 1.0

        self.unet_config_in_channels = 4
        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._execution_device = 'cpu'

    def set_height_width(self, height, width):
        pass

    def _preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB") # RGBA or other -> RGB
        image = np.array(image)
        h, w = image.shape[:-1]
        if h != 512 or w != 512:
            image = cv2.resize(
                image,
                (512,512),
                interpolation=cv2.INTER_LANCZOS4
            )
        # normalize
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        # to batch
        image = image[None].transpose(0, 3, 1, 2)
        return image
    
    def _encode_image(self, init_image):
        moments = self.vae_encoder({
            "input.1": self._preprocess_image(init_image)
        })[0]
        mean, logvar = np.split(moments, 2, axis=1)
        logvar = np.clip(logvar, -30.0, 20.0)
        std = np.exp(logvar * 0.5)
        latent = (mean + std * np.random.randn(*mean.shape)) * self.vae_config_scaling_factor
        return torch.from_numpy(latent)

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        prompt_2 = None,
        device = None,
        num_images_per_prompt = 1,
        do_classifier_free_guidance = True,
        negative_prompt = None,
        negative_prompt_2 = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        lora_scale = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = self.tokenizer
        text_encoders = self.text_encoder
        
        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = prompt

            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds_data = text_encoder({"prompt_tokens":text_input_ids.numpy()})
                
                # We are only ALWAYS interested in the pooled output of the final text encoder

                pooled_prompt_embeds = torch.tensor(prompt_embeds_data[0])
                prompt_embeds = prompt_embeds_data[1]

                prompt_embeds_list.append(torch.tensor(prompt_embeds))
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config_force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []

            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                negative_prompt_embeds = text_encoder({"prompt_tokens":uncond_input.input_ids.numpy()}) #text_encoder.run(input_node, {"prompt_tokens":uncond_input.input_ids.numpy()})
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = torch.tensor(negative_prompt_embeds[0])
                negative_prompt_embeds = negative_prompt_embeds[1]

                negative_prompt_embeds_list.append(torch.tensor(negative_prompt_embeds))

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        lora_scale = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder({"tokens": np.array(text_input_ids)})
            prompt_embeds = prompt_embeds[0]
            
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            #uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_text_input_ids = uncond_input.input_ids
            negative_prompt_embeds = self.text_encoder({"tokens": np.array(uncond_input_text_input_ids)})

            negative_prompt_embeds = negative_prompt_embeds[0]
            
        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)

        return prompt_embeds
    
    def _encode_prompt_parallel(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        lora_scale = None,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            self.text_parent_pipe.send({"state":1,"prompt":prompt})
            
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            self.text_parent_pipe2.send({"state":2,"negative_prompt":negative_prompt,
                                     "prompt":prompt,
                                     "batch_size":batch_size})
            negative_prompt_embeds = self.text_parent_pipe2.recv()
            negative_prompt_embeds = negative_prompt_embeds["from_sub"]

        if prompt_embeds is None:
            prompt_embeds = self.text_parent_pipe.recv()
            prompt_embeds = prompt_embeds["from_sub"]

    
        if do_classifier_free_guidance:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)

        return prompt_embeds
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        strength,
        num_inference_steps,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` cannot be None.")
        elif not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` has to be a positive integer but is {num_inference_steps} of type"
                f" {type(num_inference_steps)}."
            )
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return torch.tensor(timesteps), len(timesteps)

        return timesteps, num_inference_steps - t_start

    def prepare_rand_latents(self, batch_size, num_channels_latents, height, width, dtype, generator):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
        
    def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        
        batch_size = batch_size * num_images_per_prompt
        init_latents = self._encode_image(image)
        init_latents = init_latents.to(dtype)

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype
    ):
        if self.config_requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet_config_addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2_config_projection_dim
        )
        expected_add_embed_dim = self.unet_add_embedding_linear_1_in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet_config_addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet_config_addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def run_unet(self, inputs):
        inputss = []
        for i in inputs:
            inputss.append(torch.tensor(i))
        inputs1 = [ inputss[0][0].unsqueeze(0), inputss[1], inputss[2][0].unsqueeze(0)  ]
        inputs2 = [ inputss[0][1].unsqueeze(0), inputss[1], inputss[2][1].unsqueeze(0)  ]
        self.parent_pipe.send({"state":1, "data":inputs2})
        self.parent_pipe2.send({"state":1, "data":inputs1})
        res1    = self.parent_pipe2.recv()
        res2    = self.parent_pipe.recv()
        res2    = res2["from_sub"]
        res1    = res1["from_sub"]
        res     = np.concatenate([res1[0], res2[0]], axis=0)
        return [res]

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    @torch.no_grad()
    def __call__(
        self,
        prompt = None,
        prompt_2 = None,
        init_image = None,
        strength = 0.7,
        height = 512,
        width = 512,
        num_inference_steps = 50,
        denoising_start = None,
        denoising_end = None,
        guidance_scale = 5.0,
        negative_prompt = None,
        negative_prompt_2 = None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        output_type = "pil",
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        original_size = None,
        crops_coords_top_left = (0, 0),
        target_size = None,
        aesthetic_score = 6.0,
        negative_aesthetic_score = 2.5,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device


        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        start_time = time.time()
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # 4. Preprocess images
        # image = self._preprocess_image(image)
        
        # 5. Prepare timesteps and latent variables
        def denoising_value_valid(dnv):
            return type(denoising_end) == float and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if init_image is None:
            timesteps = self.scheduler.timesteps
            num_channels_latents = self.unet_config_in_channels
            latents = self.prepare_rand_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                self.prompt_embeds_dtype,
                generator,
            )
        else:
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None)

            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            add_noise = True if denoising_start is None else False
            latents = self.prepare_latents(
                init_image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                self.prompt_embeds_dtype,
                device,
                generator,
                add_noise,
            )
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        original_size = (height, width)
        target_size = (height, width)

        # 9. Denoising loop

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # 9.1 Apply denoising_end
        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
            
        for i, t in tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestamp = np.array([t])
            # predict the noise residual
            # import time
            # tm = time.time()
            if self.parallel:
                noise_pred = self.run_unet([
                    latent_model_input,
                    timestamp,
                    prompt_embeds]
                )[0]
            else:
                noise_pred = self.unet([
                    latent_model_input,
                    timestamp,
                    prompt_embeds]
                )[0]
            # print(time.time() - tm, noise_pred.shape)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.array_split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)


        latents = latents.to(torch.float32)
        if not output_type == "latent":
            init_image = torch.from_numpy(self.vae_decoder([latents / self.vae_config_scaling_factor])[0][0])
        else:
            return latents
        if len(init_image.shape) == 3:
            init_image = init_image.unsqueeze(0)
        
        # if self.parallel:
        #     self.parent_pipe.send({"state":2})
        #     self.process.join()
        #     self.parent_pipe2.send({"state":2})
        #     self.process2.join()
        #     self.text_parent_pipe.send({"state":3})
        #     self.text_process.join()
        #     self.text_parent_pipe2.send({"state":3})
        #     self.text_process2.join()
        init_image = self.image_processor.postprocess(init_image, output_type=output_type)
        return init_image[0]


if __name__ == "__main__":

    pipe = StableDiffusionPipeline(
            vae_encoder_path='/workspace/demos/onebatch/sdv1.5/models/vae_encoder_1684x_f16.bmodel', # vae_encoder_1684x_bf16.bmodel',
            vae_decoder_path='/workspace/demos/onebatch/sdv1.5/models/tae1_5.bmodel',
            text_encoder_path='/workspace/demos/onebatch/sdv1.5/models/text_encoder_1684x_f32.bmodel',
            tokenizer_path='/workspace/demos/onebatch/sdv1.5/models/tokenizer',
            unet_path='/workspace/demos/onebatch/sdv1.5/models/unet_1684x_1bs_f16.bmodel',
            parallel=True
            # unet_path="/workspace/aa/sdxl/models/unet.bmodel"
            )

    prompt = "fuzzy cat"
    negative = "strip"
    
    time1 = time.time()
    image0 = pipe(prompt,
                 negative_prompt = negative,
                 num_inference_steps = 20)
    print('Totoal time: {:4f}s '.format(time.time() - time1))
    image0.save('test_xinhong.png')
    

    # 销毁 
    # image = pipe(prompt + ", blue hair", 
    #             image=Image.open('my_eyes.png'),
    #             num_inference_steps = 20)
    # image.save('girl_img2img.png')
