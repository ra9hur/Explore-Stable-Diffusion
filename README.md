# To Explore Stable Diffusion

Here is an attempt to understand how stable diffusion works through the working code. 

![overview](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/56bfa600-5878-437d-8aaa-ab940e778e27)

Stable Diffusion is a latent text-to-image diffusion model. Here, we don't learn the distribution p(x) of dataset of images, but rather the distribution of latent representation of data.

A 512X512 sized data (image) is transformed to its latent representation which is 64X64. This allows us to reduce the computation and also the reduced number of steps to generate a sample. Also, this reduces the cost of training and inference that have the potential to democratise high resolution image synthesis to masses.


## Architecture

![Architecture](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/fc61aa3c-d739-4419-9da6-7277ad255d9c)


There are 4 main components in stable diffusion model.
1. An Autoencoder (VAE)
2. CLIP's Text Encoder
3. U-Net
4. Scheduler OR Sampler

### 1. VAE

The VAE model has two parts, an encoder and a decoder. During latent diffusion training, the encoder converts a 512*512*3 image into a low dimensional latent representation of image of size say 64*64*4 for the forward diffusion process. We call these small encoded versions of images as latents. We apply more and more noise to these latents at each step of training. This encoded latent representation of images acts as the input to the U-Net model.

Here, we are converting an image of shape (3, 512, 512) into a latent of shape(4, 64, 64), which requires 48 times less memory. This leads to reduced memory and compute requirements compared to pixel-space diffusion models. Thus, we are able to generate 512 × 512 images very quickly on 16GB Colab GPUs as well.

The decoder transforms the latent representation back into an image. We convert the denoised latents generated by the reverse diffusion process into images using the VAE decoder.

### 2. Text Encoder

he text-encoder transforms the input prompt into an embedding space that goes as input to the U-Net. This acts as guidance for noisy latents when we train Unet for its denoising process. The text encoder is usually a simple transformer-based encoder that maps a sequence of input tokens to a sequence of latent text-embeddings. Stable Diffusion does not train a new text encoder and instead uses an already trained text encoder, CLIP. The text encoder creates embeddings corresponding to the input text.

![text_embed](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/9f7506e5-3d48-44a5-9f20-fedc96c6d4bd)

For example, the text prompt a cute and adorable bunny is split into the tokens a, cute, and, adorable, and bunny, and the token sequence is padded with START and END tokens.

Each token is then converted into a vector representation containing image-related information by the text encoder of a neural network called CLIP. 


### 3. U-Net

The U-Net predicts denoised image representation of noisy latents. Here, noisy latents act as input to Unet and the output of UNet is noise in the latents. Using this, we are able to get actual latents by subtracting the noise from the noisy latents.

The Unet that takes in the noisy latents (x), timestep (t) and text embedding as guidance.

The model is essentially a UNet with an encoder(12 blocks), a middle block and a skip connected decoder(12 blocks). In these 25 blocks, 8 blocks are down sampling or upsampling convolution layer and 17 blocks are main blocks that each contain four resnet layers and two Vision Transformers(ViTs). 

Here the encoder compresses an image representation into a lower resolution image representation and the decoder decodes the lower resolution image representation back to the original higher resolution image representation that is supposedly less noisy.

### 4. Scheduler OR Sampler

Scheduler decides how much of the predicted noise to actually remove from the input image latent. Gradually removing small amounts of noise helps refine the image latents gradually and produce sharper ones.

By default, the scheduler makes this decision by accounting for the total number of timesteps. The downscaled noise is then subtracted from the latent representation of the current timestep to obtain the refined representation, which becomes the latent representation of the next timestep:

latent representation of timestep t+1 = latent representation of timestep t - downscaled noise


## Training The Diffusion Model

![Training](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/1d8d1f12-c9d2-43d8-ae66-083e7f4dce31)

Stable Diffusion is a large text-to-image diffusion model trained on billions of images. Image diffusion model learn to denoise images to generate output images. Stable Diffusion uses latent images encoded from training data as input. Further, given an image Z_o, the diffusion algorithm progressively add noise to the image and produces a noisy image Z_t, with t being how many times noise is added. When t is large enough, the image approximates pure noise. Given a set of inputs such as time step t, text prompt, image diffusion algorithms learn a network to predict the noise added to the noisy image Z_t.

The training of the Diffusion Model can be divided into two parts:

### Forward Diffusion Process → add noise to the image
The forward diffusion process gradually adds Gaussian noise to the input image X_o step by step, and there will be t steps in total. 
The process will produce a sequence of noisy image samples X_o, …, X_t.

When t tends to infinity, the final result will become a completely noisy image as if it is sampled from an isotropic Gaussian distribution.

![forward_process](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/8396d3a1-5582-4174-83c6-1d4cf9fa3362)

Referring to formula 4 above, instead of designing an algorithm to iteratively add noise to the image, a closed-form formula can be used to determine sufficient noise to move from x_0 to x_t without calculating intermediate images.

Also, since  alpha_t and bet_t are known parameters, we can conclude that there is nothing for the network to learn in the froward process.


### Reverse Diffusion Process → remove noise from the image

#### Predicting Noise

Since we start from noise in the reverse process, how can the model know the output and also the direction that it has to take to achieve ? How the model understand the prompt ?

UNet predicts a prompt-conditioned noise in the current image representation under the guidance of the text prompt's representation and timestep.

However, even though we condition the noise prediction with the text prompt, the generated image representation may not adhere strongly enough to the text prompt. To improve the adherence of the predicted noise to the text prompt, Stable Diffusion additionally predicts generic noise conditioned on an empty prompt (" "). The final noise prediction is a weighted sum of the predicted generic noise and the prompt-conditioned noise with the weights controlled by the hyperparameter guidance scale:

output = guid_scale * (output_cond - output_uncond) + output_uncond

A guidance scale of 0 means no adherence to the text prompt, while a guidance scale of 1 means using only the prompt-conditioned noise without introducing generic noise. Larger guidance scales result in stronger adherence to the text prompt. To see how the introduction of generic noise and guidance scale enhances image quality, you can check out our Diffusion Explainer by setting the guidance scale to 0 or 7.


#### Predicting Noisy Image

![reverse_process](https://github.com/ra9hur/Explore-Stable-Diffusion/assets/17127066/f28648ed-32c2-4dfb-a221-296b4185c062)

UNET learns to predict noise (epsilon_not). Hence, the above formula can be used to predict X_t-1.




## References

1. [Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

2. [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

3. [Paper: High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

2. [Stability-AI/stable diffusion](https://github.com/Stability-AI/stablediffusion)

3. [Stable Diffusion Explained Step-by-Step with Visualization](https://medium.com/polo-club-of-data-science/stable-diffusion-explained-for-everyone-77b53f4f1c4)

4. [Paper: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

5. [Stable Diffusion Explained](https://medium.com/@onkarmishra/stable-diffusion-explained-1f101284484d)

6. [Stable Diffusion - What, Why, How?](https://www.youtube.com/watch?v=ltLNYA3lWAQ)