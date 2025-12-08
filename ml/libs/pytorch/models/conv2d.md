# Conv2d

- in_channels: int - число входных каналов. Его нужно правильно прописать. Соответствует числу каналов изображения
- out_channels: int - число выходных каналов
- kernel_size: int | tuple[int, int] - размер ядра либо сторона квадрата либо матрица
- stride: int | tuple[int, int] = 1 - шаг
- padding: str | int | tuple[int, int] = 0 - отступ, valid == 0, same - для регулирования выходов
- dilation: int | tuple[int, int] = 1 разрыв между элементами ядра
- groups: int = 1 - число заблокированных соединений
- bias: bool = True - байес
- padding_mode: str = "zeros" - так можно задать нулевой отступ 'reflect', 'replicate','circular'
- device: Any = None - устройство
- dtype: Any = None - тип

Выходные значения буду пересчитываться по формуле

```python
x = lambda input_size_x, padding, kernel_size, stride, dilation: (
        (input_size_x + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1) 
```

Вычисление размерности выходного тензора выходного тензора

```python
import math


def conv2d_output_shape(H_in, W_in, kernel_size, stride, padding):
    """
    Вычисляет размеры выхода Conv2d.
    
    H_in, W_in : int
        Размеры входного изображения (высота, ширина)
    kernel_size : (int, int)
        Размер ядра (kernel_H, kernel_W)
    stride : (int, int)
        Шаг свёртки (stride_H, stride_W)
    padding : (int, int)
        Паддинг (pad_H, pad_W)
    """
    kernel_H, kernel_W = kernel_size
    stride_H, stride_W = stride
    pad_H, pad_W = padding

    H_out = math.floor((H_in + 2 * pad_H - kernel_H) / stride_H) + 1
    W_out = math.floor((W_in + 2 * pad_W - kernel_W) / stride_W) + 1

    return H_out, W_out


# проверим на твоём примере:
H, W = 17, 19
kernel_size = (5, 5)
stride = (1, 1)
padding = (2, 2)

print(conv2d_output_shape(H, W, kernel_size, stride, padding))
# 👉 должно дать (17, 19)

```

Реализация прохода ядер слоев

```python
import torch

import torch

C = 3  # число каналов
H, W = 16, 12  # размеры изображения: H - число строк; W - число столбцов
kernel_size = (5, 3)  # размер ядра по осям (H, W)
stride = (1, 2)  # шаг смещения ядра по осям (H, W)
padding = 1  # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

bias = torch.rand(1)  # смещение для фильтра (ядра), коэффициент w0
act = torch.tanh  # функция активации нейронов (результатов свертки)

H_out = int((H + 2 * padding - kernel_size[0]) / stride[0] + 1)
W_out = int((W + 2 * padding - kernel_size[1]) / stride[1] + 1)

x_img = torch.randint(0, 255, (C, H, W), dtype=torch.float32)  # тензоры x_img и kernel
kernel = torch.rand((C,) + kernel_size)  # 3 слоя, 5 * 3 размер одного фильтра

# здесь продолжайте программу
predict = torch.zeros(H_out, W_out, dtype=torch.float32)

# добавление отступов нативная реализация
# x_img_zeros = torch.zeros((C, H + 2 * padding, W + 2 * padding))
# x_img_zeros[:, padding:-padding, padding:-padding] = x_img

# добавление отступов через torch.nn.functional
x_img_pad = torch.nn.functional.pad(x_img, (padding, padding, padding, padding))

for i in range(H_out):
    for j in range(W_out):
        predict[i, j] = torch.sum(
            x_img_pad[
            :,  # весь батч
            i * stride[0]:kernel_size[0] + i * stride[0],  # по x
            j * stride[1]:kernel_size[1] + j * stride[1],  # по y
            ] * kernel  # помножаем на ядро
        )

predict = act(predict + bias)
```