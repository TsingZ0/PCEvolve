import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration


class BlipWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.server_captioner == 'BlipBase':
            model_dir = "captioner/blip/base"
            self.processor = BlipProcessor.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True, 
                do_rescale=False
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True
            )
        elif args.server_captioner == 'BlipLarge':
            model_dir = "captioner/blip/large"
            self.processor = BlipProcessor.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True, 
                do_rescale=False
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True
            )
        elif args.server_captioner == 'Blip2':
            model_dir = "captioner/blip2-opt-2.7b"
            self.processor = Blip2Processor.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True, 
                do_rescale=False
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True, 
                device_map='auto'
            )
        else:
            raise NotImplementedError

        self.model.to(args.device)

    def __call__(self, image):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors='pt').to(self.args.device)
            out = self.model.generate(**inputs, max_length=self.args.caption_max_length)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return text

class LLaVAWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.server_captioner == 'LLaVA':
            model_dir = "captioner/llava-1.5-7b-hf"
            self.processor = AutoProcessor.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
                do_rescale=False
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
                device_map='auto'
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(args.device)

    def __call__(self, image):
        with torch.no_grad():
            prompt = "USER: <image>\nWhat's the details of the image? ASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.args.device)
            out = self.model.generate(**inputs, max_length=self.args.caption_max_length)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()[len(prompt)-len('image'):]
            return text
    

def get_captioner(args):
    if 'Blip' in args.server_captioner:
        return BlipWrapper(args)
    elif 'LLaVA' in args.server_captioner:
        return LLaVAWrapper(args)
    else:
        raise NotImplementedError