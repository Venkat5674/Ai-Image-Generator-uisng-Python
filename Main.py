!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A futuristic city at night with towering skyscrapers, neon lights in shades of blue and pink, flying cars, and people walking on a glowing transparent bridge, in a cyberpunk art style, with a dramatic and moody atmosphere"


image = pipe(prompt, width=1000, height=1000).images[0]

plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
