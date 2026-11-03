import torch
import torch.nn as nn

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def conv_block(in_ch: int, out_ch: int, p_drop: float) -> nn.Sequential:
    """
    order : conv -> Batch normalization -> Ativation using ReLU -> pool -> drop
    :param in_ch: Number of channels in the input image
    :param out_ch: Number of channels produced by the convolution
    :param p_drop: probability of an element to be zeroed
    :return: self
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(p_drop),
    )

class CNN(nn.Module):
    def __init__(self,num_classes: int = 10, p_drop:float = 0.2, hidden: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,32,p_drop),     # 32x32 -> 16x16
            conv_block(32,64,p_drop),    # 16x16 -> 8x8
            conv_block(64,128,p_drop),   # 8x8x -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)