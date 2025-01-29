import torchvision
import torch.nn as nn

class ViTWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.client_model == 'ViT-B16' or args.client_use_embedding == 'ViT-B16':
            # image min_size: height=224, width=224
            self.model = torchvision.models.vit_b_16(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 768
        elif args.client_model == 'ViT-B32' or args.client_use_embedding == 'ViT-B32':
            # image min_size: height=224, width=224
            self.model = torchvision.models.vit_b_32(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 768
        elif args.client_model == 'ViT-L16' or args.client_use_embedding == 'ViT-L16':
            # image min_size: height=224, width=224
            self.model = torchvision.models.vit_l_16(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 1024
        elif args.client_model == 'ViT-L32' or args.client_use_embedding == 'ViT-L32':
            # image min_size: height=224, width=224
            self.model = torchvision.models.vit_l_32(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 1024
        else:
            raise NotImplementedError
        
        self.model.heads = nn.Identity()
        self.model.to(args.device)

        self.resize = torchvision.transforms.Resize(size=224)

    def forward(self, x):
        x = self.resize(x)
        out = self.model(x)
        return out


def get_vit(args):
    return ViTWrapper(args)