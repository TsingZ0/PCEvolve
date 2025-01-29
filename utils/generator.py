import base64
import random
import requests
import torch
import inspect
import logging
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from io import BytesIO
from PIL import Image

module = inspect.getmodule(StableDiffusionSafetyChecker)
logging.getLogger(module.__name__).setLevel(logging.ERROR)

# get your own AUTHORIZATIONKEY from https://cloud.siliconflow.cn/account/ak
# access to the siliconflow API is free
AUTHORIZATIONKEY = ""


class Text2ImageWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.online_api:
            self.AuthorizationKey = AUTHORIZATIONKEY
            if args.server_generator == 'StableDiffusion-2-1':
                self.api_model_name = 'stabilityai/stable-diffusion-2-1'
            else:
                raise NotImplementedError
        else:
            if args.server_generator == 'StableDiffusion':
                self.GenPipe = AutoPipelineForText2Image.from_pretrained(
                    'generator/stable-diffusion-v1-5', 
                    torch_dtype=torch.float16, 
                    device_map='balanced', 
                    local_files_only=True, 
                    use_safetensors=True,
                )
            else:
                raise NotImplementedError
            
            self.GenPipe.set_progress_bar_config(disable=True)

        self.img_size = max(512, args.img_size)


    def __call__(self, prompt, negative_prompt):
        if self.args.online_api:
            url = "https://api.siliconflow.cn/v1/images/generations"
            payload = {
                "model": self.api_model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": f"{self.img_size}x{self.img_size}",
                "batch_size": 1,
                "seed": random.randint(1, 4999999999),
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
            headers = {
                "Authorization": f"Bearer {self.AuthorizationKey}",
                "Content-Type": "application/json"
            }
            while True:
                response = requests.request("POST", url, json=payload, headers=headers)
                if response.status_code == 200:
                    generated_images = []
                    image_urls = []
                    for image in response.json()["images"]:
                        image_url = image["url"]
                        image_urls.append(image_url)
                        generated_content = requests.get(image_url).content
                        generated_bytes = BytesIO(generated_content)
                        generated_image = Image.open(generated_bytes)
                        generated_images.append(generated_image)
                    return generated_images, image_urls
                else:
                    print('Error: response.status_code:', response.status_code)
                    print(response.text)
        else:
            with torch.no_grad():
                res = self.GenPipe(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                )

                generated_images = []
                for idx, nsfw in enumerate(res.nsfw_content_detected):
                    if not nsfw:
                        generated_images.append(res.images[idx])
                return generated_images, []


class Image2ImageWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.online_api:
            self.AuthorizationKey = AUTHORIZATIONKEY
            if args.server_generator == 'StableDiffusion-2-1':
                self.api_model_name = 'stabilityai/stable-diffusion-2-1'
            else:
                raise NotImplementedError
        else:
            if args.server_generator == 'StableDiffusion':
                self.GenPipe = AutoPipelineForImage2Image.from_pretrained(
                    'generator/stable-diffusion-v1-5', 
                    torch_dtype=torch.float16, 
                    device_map='balanced', 
                    local_files_only=True, 
                    use_safetensors=True,
                )
                if args.use_IPAdapter:
                    self.GenPipe.load_ip_adapter(
                        "generator/IP-Adapter", 
                        subfolder="models", 
                        weight_name="ip-adapter_sd15.safetensors"
                    )
            else:
                raise NotImplementedError
            
            self.GenPipe.set_progress_bar_config(disable=True)
            if args.use_IPAdapter:
                self.GenPipe.set_ip_adapter_scale(args.IPAdapter_scale)

        self.img_size = max(512, args.img_size)


    def __call__(self, prompt, img, negative_prompt):
        if self.args.online_api:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            url = "https://api.siliconflow.cn/v1/images/generations"
            payload = {
                "model": self.api_model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": f"{self.img_size}x{self.img_size}",
                "batch_size": 1,
                "seed": random.randint(1, 4999999999),
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "image": image_base64, 
            }
            headers = {
                "Authorization": f"Bearer {self.AuthorizationKey}",
                "Content-Type": "application/json"
            }
            while True:
                response = requests.request("POST", url, json=payload, headers=headers)
                if response.status_code == 200:
                    generated_images = []
                    image_urls = []
                    for image in response.json()["images"]:
                        image_url = image["url"]
                        image_urls.append(image_url)
                        generated_content = requests.get(image_url).content
                        generated_bytes = BytesIO(generated_content)
                        generated_image = Image.open(generated_bytes)
                        generated_images.append(generated_image)
                    return generated_images, image_urls
                else:
                    print('Error: response.status_code:', response.status_code)
                    print(response.text)
        else:
            with torch.no_grad():
                image = img.resize((self.img_size, self.img_size))
                res = self.GenPipe(prompt=prompt, 
                    image=image, 
                    strength=self.args.i2i_strength, 
                    negative_prompt=negative_prompt, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                    ip_adapter_image=image if self.args.use_IPAdapter else None, 
                )

                generated_images = []
                for idx, nsfw in enumerate(res.nsfw_content_detected):
                    if not nsfw:
                        generated_images.append(res.images[idx])
                return generated_images, []


def get_generator(args):
    if args.task_mode == 'T2I':
        return Text2ImageWrapper(args)
    elif args.task_mode == 'I2I':
        if args.random_gen:
            return Text2ImageWrapper(args)
        else:
            return Image2ImageWrapper(args)
    else:
        raise NotImplementedError