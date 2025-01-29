import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection


class CLIPImageWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        model_dir = "clip/openai/clip-vit-large-patch14"
        self.processor = AutoProcessor.from_pretrained(
            model_dir, 
            # torch_dtype=torch.float16, 
            local_files_only=True, 
            do_rescale=False, 
            do_resize=True
        )
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_dir, 
            # torch_dtype=torch.float16, 
            local_files_only=True
        )
        self.feature_dim = 768
        
        self.model.to(args.device)

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.args.device)
        image_embeds = self.model(**inputs).image_embeds
        return image_embeds


def get_clip(args):
    return CLIPImageWrapper(args)