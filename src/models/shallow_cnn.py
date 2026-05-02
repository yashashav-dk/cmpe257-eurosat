"""ShallowCNN ~391K params with toggleable BN and Dropout."""
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self, num_classes=10, use_batchnorm=False, dropout_rate=0.0):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        def block(in_ch, out_ch, last=False):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if not last:
                layers.append(nn.MaxPool2d(2, 2))
                if dropout_rate > 0:
                    layers.append(nn.Dropout2d(dropout_rate))
            return nn.Sequential(*layers)

        self.block1 = block(3, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.block4 = block(128, 256, last=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_fc = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
