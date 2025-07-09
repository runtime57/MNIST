import torchvision
from torch import nn


class ResNetModel(nn.Module):
    def __init__(self, input_channels, n_class, use_pretrained):

        super.__init__()

        if use_pretrained:
            weights = torchvision.models.ResNet18_Weights
        else:
            weights = None
        self.resnet = torchvision.models.resnet18(weights=weights)
        self.resnet.conv1 = nn.Conv2d(
            input_channels,
            self.resnet.conv1.out_channels,
            self.resnet.conv1.kernel_size,
            self.resnet.conv1.stride,
            self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_class)

    def forward(self, img, **batch):
        return {"logits": self.resnet(img)}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
