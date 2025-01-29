import torchvision
import torch.nn as nn


class InceptionWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = torchvision.models.inception_v3(
            torchvision.models.Inception_V3_Weights.DEFAULT
        )
        self.feature_dim = 2048
        self.model.fc = nn.Identity()
        self.model.to(args.device)

        self.resize = torchvision.transforms.Resize(size=299)

    def forward(self, x):
        x = self.resize(x)
        out = self.model(x).logits
        return out


def get_inception(args):
    return InceptionWrapper(args)