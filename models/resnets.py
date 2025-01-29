import torchvision
import torch.nn as nn

class ResNetWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.client_model == 'ResNet18' or args.client_use_embedding == 'ResNet18':
            self.model = torchvision.models.resnet18(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 512
        elif args.client_model == 'ResNet34' or args.client_use_embedding == 'ResNet34':
            self.model = torchvision.models.resnet34(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 512
        elif args.client_model == 'ResNet50' or args.client_use_embedding == 'ResNet50':
            self.model = torchvision.models.resnet50(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 2048
        elif args.client_model == 'ResNet101' or args.client_use_embedding == 'ResNet101':
            self.model = torchvision.models.resnet101(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 2048
        elif args.client_model == 'ResNet152' or args.client_use_embedding == 'ResNet152':
            self.model = torchvision.models.resnet152(
                pretrained=args.client_model_pretrained
            )
            self.feature_dim = 2048
        else:
            raise NotImplementedError
            
        self.model.fc = nn.Identity()
        self.model.to(args.device)

    def forward(self, x):
        out = self.model(x)
        return out


def get_resnet(args):
    return ResNetWrapper(args)