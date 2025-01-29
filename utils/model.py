import torch.nn as nn
import torch
from models.clip import get_clip
from models.inception import get_inception
from models.vits import get_vit
from models.resnets import get_resnet


class ModelWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if 'CLIP' in args.client_model:
            assert args.client_model_pretrained==True 
            self.encoder = get_clip(args)
        elif 'Inception' in args.client_model:
            assert args.client_model_pretrained==True
            self.encoder = get_inception(args)
        elif 'ViT' in args.client_model:
            self.encoder = get_vit(args)
        elif 'ResNet' in args.client_model:
            self.encoder = get_resnet(args)
        else:
            raise NotImplementedError
        
        self.head = nn.Linear(self.encoder.feature_dim, args.num_labels, bias=False)
        self.head.to(args.device)

    def __call__(self, x):
        if self.args.client_encoder_fixed:
            with torch.no_grad():
                feature = self.encoder(x).detach()
        else:
            feature = self.encoder(x)
        output = self.head(feature)
        return output


def get_model(args):
    return ModelWrapper(args)