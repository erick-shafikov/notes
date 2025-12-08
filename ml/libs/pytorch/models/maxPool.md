# Нативная реализация

# Max pooling

```python
import torch

H, W = 16, 12  # размеры карты признаков: H - число строк; W - число столбцов
kernel_size = (2, 2)  # размер окна для Pooling по осям (H, W)
stride = (2, 2)  # шаг смещения окна по осям (H, W)
padding = 0  # размер нулевой области вокруг карты признаков (число строк и столбцов с каждой стороны)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x = torch.rand((H, W))  # карта признаков (в программе не менять)

# здесь продолжайте программу
res_pool = torch.zeros((H_out, W_out))

for i in range(H_out):
    for j in range(W_out):
        stride_h, stride_w = stride
        kernel_h, kernel_w = kernel_size

        res_pool[i, j] = torch.max(
            x[i * stride_h: i * stride_h + kernel_h, j * stride_w: j * stride_w + kernel_w])
```

# Average pooling

```python
import torch

H, W = 24, 24  # размеры карты признаков: H - число строк; W - число столбцов
kernel_size = (3, 2)  # размер окна для Pooling по осям (H, W)
stride = (2, 2)  # шаг смещения окна по осям (H, W)
padding = 1  # размер нулевой области вокруг карты признаков (число строк и столбцов с каждой стороны)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x = torch.rand((H, W))  # карта признаков (в программе не менять)

# здесь продолжайте программу
x_padding = torch.nn.functional.pad(x, (padding, padding, padding, padding))

res_pool = torch.zeros((H_out, W_out))

stride_h, stride_w = stride
kernel_h, kernel_w = kernel_size

for i in range(H_out):
    for j in range(W_out):
        res_pool[i, j] = torch.mean(
            x_padding[i * stride_h: i * stride_h + kernel_h, j * stride_w: j * stride_w + kernel_w])

```