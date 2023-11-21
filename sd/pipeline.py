import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt=None,     # negative prompt OR empty string
    input_image=None,
    strength=0.8,
    do_cfg=True,            # classifier free guidance
    # weight of how much we want the model to pay attention to our prompt
    cfg_scale=7.5,          # it's a value that goes from 1 to 14  
    sampler_name='ddpm',
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        # if you have a very limited GPU and you want to offload the models after using them, 
        # you can offload them back to the CPU by moving them to the CPU
        if idle_device:
            to_idle = lambda x: x.to(device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed in None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]       # pre-trained model
        clip.to(device)

        # If 'yes', consider both conditional and unconditional prompts
        if do_cfg:
            # Convert into a list of length seq_len=77
            # Fill the  remaining space with padding if  the  prompt is small
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77).input_ids
            
            # Convert input_ids to torch tensor
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # Input IDs are converted to embeddings
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)   # dim - 768 sized vector
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77).input_ids
            
            # (batch_size, seq_len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)

            # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) 
            # -> (2 * batch_size, seq_len, dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length seq_len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77).input_ids
            
            # Convert to torch tensor
            # (batch_size, seq_len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            context = clip(tokens)
        to_idle(clip)


        # Other sampler alternatives - ddim
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)


        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            
            # (h, w, c)
            input_image_tensor = np.array(input_image_tensor)

            # Convert to torch tensor
            # (h, w, c) -> (h, w, c)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (h, w, c) -> (batch, h, w, c)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # (batch, h, w, c) -> (batch, c, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (batch, 4, lat_h, lat_w)
            encoder_noise = torch.rand(latents_shape, generator=generator, device=device)
            # (batch, 4, lat_h, lat_w)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents ( the encoded image)
            # (batch, 4, lat_h, lat_w)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Depending on the strength, more the noise added, UNET would have more flexibility
            # and output image will be much different

            to_idle(encoder)

        else:
            # If it is text-to-image, start with random noise N(0, I)
            # (batch, 4, lat_h, lat_w)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # timesteps - (1000, 950, 900, 850 ... 0 - 50 time steps)
        # Each time step represents a noise level
        # Means we can instruct schedular to remove the extent of noise based on timestep
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch, 4, lat_h, lat_w) OR (batch, 4, 64, 64)
            model_input = latents

            if do_cfg:
                # Latent repeated twice for conditional and unconditional inputs
                # one used with the prompt and another without prompt
                # (batch, 4, lat_h, lat_w) -> (2 * batch, 4, lat_h, lat_w)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (batch, 4, lat_h, lat_w)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Scheduler removes predicted noise (model_output) from the input image
            # This image (latent) is again used as input in the next time-step
            # (batch, 4, lat_h, lat_w)
            latents = sampler.step(timestep, latents, model_output)


        to_idle(diffusion)


        decoder = models["decoder"]
        decoder.to(device)

        # (batch, 4, h, w) -> (batch, 3, h, w)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (batch, c, h, w) -> (batch, h, w, c)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]



def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


# Function exactly same as in transformers for positional embeddings
def get_time_embedding(timestep):
    
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

