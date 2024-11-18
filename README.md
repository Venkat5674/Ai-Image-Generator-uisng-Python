🖼️ AI Image Generator using Stable Diffusion in Google Colab
This project demonstrates the power of AI in generating visually stunning and imaginative images using Stable Diffusion. With Python and Google Colab, you can create unique AI-generated art simply by crafting a creative prompt.

🚀 Features
Generate Stunning Images: Create high-quality images based on textual descriptions.
GPU-Accelerated: Leverages CUDA for faster image generation.
Customizable Prompts: Adjust the description to generate various styles and themes.

📋 Requirements
Before you begin, ensure you have:
Google Colab (preferred)
An active NVIDIA GPU driver if running locally
Python libraries as specified below

🛠️ Installation
1.Clone this repository or copy the code into your environment.
2.Install the required dependencies by running:
!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers

📝 Steps to Generate Images
Step 1: Import Dependencies
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

Step 2: Clear GPU Cache
torch.cuda.empty_cache()

Step 3: Load Stable Diffusion Model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

Step 4: Define the Prompt
Set up a detailed and descriptive prompt for your image. For example:
prompt = "A futuristic city at night with towering skyscrapers, neon lights in shades of blue and pink, flying cars, and people walking on a glowing transparent bridge, in a cyberpunk art style, with a dramatic and moody atmosphere"

Step 5: Generate and Display the Image
image = pipe(prompt, width=1000, height=1000).images[0]
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

🌟 Example Output
The above code will generate a cyberpunk-style futuristic cityscape with vibrant neon lighting, flying cars, and a moody atmosphere.

💡 Customize Your Art
Change the prompt to explore various styles and scenes.
Experiment with image resolution by modifying the width and height parameters.
Try different Stable Diffusion models for diverse results.

📂 Resources
Stable Diffusion Documentation:[https://stablediffusionapi.com/docs/]
GoogleColab:[https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.01-Help-And-Documentation.ipynb]

🌐 License
This project is licensed under the MIT License. Feel free to use, modify, and share.

🤝 Contributing
Have ideas or improvements? Contributions are welcome!

🔗 Connect
For any queries or feedback, feel free to reach out. Let’s explore AI creativity together!

Enjoy creating your AI-generated masterpieces! 🎨

![WhatsApp Image 2024-11-18 at 18 24 35_79d3811d](https://github.com/user-attachments/assets/25ee9c46-c034-4158-b3e1-e328b0eae017)

