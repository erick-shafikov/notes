# Basic block

реализация Basic block

```python
import torch
import torch.nn as nn

# здесь объявляйте класс модели

batch_size = 8
x = torch.rand(batch_size, 64, 32, 32)  # тензор x в программе не менять


# здесь продолжайте программу
class BasicBloc(nn.Module):
    def __init__(self):
        super().__init__()

        self.x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_1 = x.clone()

        x = self.x(x)
        x = self.skip_connection(x)

        return x_1 + x


model_bb = BasicBloc()

model_bb.eval()
y = model_bb(x)
```

# Bottleneck

реализация Bottleneck block

```python
import torch
import torch.nn as nn

# здесь объявляйте класс модели

batch_size = 4
x = torch.rand(batch_size, 256, 16, 16)  # тензор x в программе не менять


# здесь продолжайте программу
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.fx = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                bias=False,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
        )

        self.out = nn.ReLU()

    def forward(self, x):
        return self.out(self.fx(x) + x)


model_bn = Bottleneck()

model_bn.eval()
y = model_bn(x)
```

# Bottleneck + skip connection

```python
import torch
import torch.nn as nn

# здесь объявляйте класс модели

batch_size = 4
x = torch.rand(batch_size, 128, 16, 16)  # тензор x в программе не менять


# здесь продолжайте программу
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.fx = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
        )

        self.side_chain = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(256)
        )

        self.out = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.out(self.fx(x) + self.side_chain(x))


model_bn = Bottleneck()

model_bn.eval()
y = model_bn(x)
```