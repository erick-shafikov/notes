## Проблема затухания градиентов и идея остаточных связей

При обучении глубоких сетей (более ~50 слоёв) возникает парадокс: добавление слоёв не улучшает, а ухудшает качество или останавливает прогресс вовсе. Причина — градиент затухает до нуля в нижних слоях при обратном распространении, и они перестают обновляться.

Решение предложили в ResNet: ввести **остаточную (skip) связь**, которая перебрасывает вход блока напрямую к его выходу:

$$h(x) = f(x) + x$$

где $f(x)$ — преобразование внутри блока (свёртки, нормализация, активация), $x$ — идентичная копия входа. Градиент по входу тогда:

$$\frac{\partial h(x)}{\partial x} = 1 + \frac{\partial f(x)}{\partial x}$$

Слагаемое $1$ гарантирует, что градиент не обнуляется даже при $\frac{\partial f}{\partial x} \to 0$. Сеть фактически обучает не полное преобразование, а **остаток** (residual): $f(x) = h(x) - x$ — поправку к тождественному отображению. Если оптимальное преобразование близко к тождественному, блоку достаточно обнулить $f$, что значительно проще, чем явно выучить тождественную функцию через стек нелинейных слоёв.

### BasicBlock (ResNet-18, ResNet-34)

Стандартный блок содержит два свёрточных слоя $3\times3$:

$$\text{Conv}_{3\times3} \to \text{BN} \to \text{ReLU} \to \text{Conv}_{3\times3} \to \text{BN} \xrightarrow{+\,x} \text{ReLU}$$

Когда число каналов или пространственный размер меняются (первый блок слоя), боковая ветвь вместо простого копирования применяет свёртку $1\times1$ с нужным шагом, чтобы привести размерность $x$ к размерности $f(x)$.

### Bottleneck (ResNet-50, ResNet-101, ResNet-152)

Для глубоких сетей (100+ слоёв) используют блок-«бутылочное горлышко»: сначала свёртка $1\times1$ **сжимает** число каналов в 4 раза, затем идёт основная свёртка $3\times3$, потом свёртка $1\times1$ **разворачивает** каналы обратно:

$$\text{Conv}_{1\times1}(C/4) \to \text{BN} \to \text{ReLU} \to \text{Conv}_{3\times3}(C/4) \to \text{BN} \to \text{ReLU} \to \text{Conv}_{1\times1}(C) \to \text{BN} \xrightarrow{+\,x} \text{ReLU}$$

где $C$ — число выходных каналов блока. Это снижает вычислительную стоимость при сохранении глубины сети.

### ResNet как ансамбль путей

Если развернуть ResNet с $n$ блоками в схему всех возможных маршрутов сигнала, обнаруживается $2^n$ различных путей от входа к выходу: с любым подмножеством блоков или без них. Это позволяет провести параллель с ансамблевыми методами и алгоритмом Dropout — ResNet неявно обучает ансамбль подсетей разной глубины. Промежуточные блоки выступают в роли «усилителей» остатка: они добавляют уточняющую коррекцию к тому, что уже передаётся по прямому пути. Именно поэтому увеличение числа блоков не приводит к деградации обучения — каждый новый блок может просто выучить нулевой остаток, не мешая остальным.

---

# resnet18

Основные моделей отличия это блоки с 4 по 7:

- resnet18, resnet34 состоят из BasicBlock
- resnet50, resnet101 из Bottleneck

Структура:

- входной слой общий для всех видоа:
  -- Conv2d(3, 64, 7x7, s=2, p=3, b=False)
  -- BatchNorm(64)
  -- MaxPool2d(3x3, s=2, p=1)
- Основное ядро которым различаются (количество в resnet18, resnet34, resnet50, resnet101)
  -- Layer1(64, H, W) (2, 3, 3, 3)
  -- Layer2(128, H/2, W/2) (2, 4, 4, 4)
  -- Layer3(256, H/4, W/4) (2, 6, 6, 23)
  -- Layer4(512, H/8, W/8) (2, 3, 3, 3)
- выходной слой - общий для всех
  -- AdaptiveAvgPool2d() - усредняет, 512 каналов, h/8, w/8 разрешение - формирует для каждого канала одно значение 1х1
  -- Linear(512\*factor, out=1000, b=True) - factor == 1 в случае resnet18

## Layer 1

BasicBlock_1(64, h, w | 64, h, w):

- Основной путь:
  -- Conv2d(64, 3x3, s=1, p=1, b=False)
  -- BatchNorm(64)
  -- ReLU()
  -- Conv2d(64, 3x3, s=1, p=1, b=False)
  -- BatchNorm(64)
- Побочный:
  -- копия
- сумма
- Relu()

BasicBlock_2(64, h, w | 64, h, w) - точная копия BasicBlock_1 Layer 1

## Layer 2

Состоит из двух BasicBlock, сигнал увеличивает число каналов

BasicBlock_1(64, h, w | 128, h/2, w/2):

- Основной путь:
  -- Conv2d(128, 3x3, s=2, p=1, b=False) - из-за s=2 уменьшается в 2 раза
  -- BatchNorm(128)
  -- Conv2d(128, 3x3, s=1, p=1, b=False )
  -- BatchNorm(128)
- Побочный:
  -- Conv2d(128, 1x1, s=2, p=0, b=False) - так как входной 64
  -- BatchNorm(128)
- сумма
- Relu()

BasicBlock_2(128, h/2, w/2 | 128, h/2, w/2):

- Основной путь:
  -- Conv2d: 128, 3x3, s=1, p=1, b=False
  -- BatchNorm: 128
  -- ReLU
  -- Conv2d: 128, 3x3, s=1, p=1, b=False
  -- BatchNorm: 128
  -- сумма
  -- ReLU
- Побочный:
  -- копия

## Layer 3

BasicBlock_1(128, h/2, w/2 | 128, h/4, w/4):

- Основной путь:
  -- Conv2d(256, 3x3, s=2, p=1, b=False)
  -- BatchNorm(256)
  -- Relu()
  -- Conv2d(256, 3x3, s=1, p=1, b=False )
  -- BatchNorm(256)
- Побочный:
  -- Conv2d(256, 1x1, s=2, p=0, b=False)
  -- BatchNorm(256)
- сумма
- Relu()

BasicBlock_2(256, h/4, w/4 | 128, h/4, w/4):

- Основной путь:
  -- Conv2d: 128, 3x3, s=1, p=1, b=False
  -- BatchNorm: 128
  -- ReLU
  -- Conv2d: 128, 3x3, s=1, p=1, b=False
  -- BatchNorm: 128
  -- сумма
  -- ReLU
- Побочный:
  -- копия

## Layer 4

BasicBlock_1(256, h/4, w/4 | 512, h/8, w/8):

- Основной путь:
  -- Conv2d(512, 3x3, s=2, p=1, b=False)
  -- BatchNorm(512)
  -- Relu()
  -- Conv2d(512, 3x3, s=1, p=1, b=False )
  -- BatchNorm(512)
- Побочный:
  -- Conv2d(512, 1x1, s=2, p=0, b=False)
  -- BatchNorm(512)
- сумма
- Relu()

BasicBlock_2(256, h/8, w/8 | 512, h/8, w/8):

- Основной путь:
  -- Conv2d(512, 3x3, s=2, p=1, b=False)
  -- BatchNorm(512)
  -- Relu()
  -- Conv2d(512, 3x3, s=1, p=1, b=False )
  -- BatchNorm(512)
- Побочный:
  -- копия
- сумма
- Relu()

# resnet50

factor == 4

## Layer 1

Bottleneck_1(64, h, w | 64 \* 4, h, w):

- Основной путь:
  -- Conv2d(64, 1x1, s=1, p=0, b=False)
  -- BatchNorm(64)
  -- ReLU()
  -- Conv2d(64, 3x3, s=1, p=1, b=False)
  -- BatchNorm(64)
  -- ReLU()
  -- Conv2d(64 _ 4, 1x1, s=1, p=0, b=False)
  -- BatchNorm(64 _ 4)
- Побочный:
  -- Conv2d(64 _ 4, 1x1, s=1, p=0, b=False)
  -- BatchNorm(64 _ 4)
- сумма
- Relu()

Bottleneck*2(64 * 4, h, w | 64 \_ 4, h, w)

- Основной путь:
  -- Conv2d(64, 1x1, s=1, p=0, b=False)
  -- BatchNorm(64)
  -- ReLU()
  -- Conv2d(64, 3x3, s=1, p=1, b=False)
  -- BatchNorm(64)
  -- ReLU()
  -- Conv2d(64 _ 4, 1x1, s=1, p=0, b=False)
  -- BatchNorm(64 _ 4)
- Побочный:
  -- копия
- сумма
- Relu()
-

## Layer 2

Bottleneck*1(64 * 4, h, w | 128 \_ 4, h/2, w/2):

- Основной путь:
  -- Conv2d(128, 1x1, s=1, p=0, b=False)
  -- BatchNorm(128)
  -- ReLU()
  -- Conv2d(128, 3x3, s=2, p=1, b=False)
  -- BatchNorm(128)
  -- ReLU()
  -- Conv2d(128 _ 4, 1x1, s=1, p=0, b=False)
  -- BatchNorm(128 _ 4)
- Побочный:
  -- Conv2d(128 _ 4, 1x1, s=2, p=0, b=False)
  -- BatchNorm(128 _ 4)
- сумма
- Relu()

Bottleneck*2, Bottleneck_3, Bottleneck_4(128 * 4, h/2, w/2 | 128 \_ 4, h/2, w/2):

- Основной путь:
  -- Conv2d(128, 1x1, s=1, p=0, b=False)
  -- BatchNorm(128)
  -- ReLU()
  -- Conv2d(128, 3x3, s=2, p=1, b=False)
  -- BatchNorm(128)
  -- ReLU()
  -- Conv2d(128 _ 4, 1x1, s=1, p=0, b=False)
  -- BatchNorm(128 _ 4)
- Побочный:
  -- копия
- сумма
- Relu()

## Layer 3

Аналогично Layer 2

Bottleneck*1(128 * 4, h/2, w/2 | 256 \_ 4, h/4, w/4)
Bottleneck_2-6(256, h/4, w/4 | 256 \* 4, h/4, w/4)

## Layer 3

Аналогично Layer 2, Layer 3

Bottleneck*1(512, h/2, w/2 | 512 * 4, h/8, w/8)
Bottleneck*2-6(512 * 4, h/8, w/8 | 512 \* 4, h/8, w/8)

# использование

```python
from PIL import Image
from torchvision import models
import torch

# Загрузка пред-обученных весов и метаданных
resnet_weights = models.ResNet50_Weights.DEFAULT
# список категорий
categories = resnet_weights.meta['categories']
# преобразования
transforms = resnet_weights.transforms()

# Создание модели с пред-обученными весами
model = models.resnet50(weights=resnet_weights)

# Загрузка и преобразование изображения
img = Image.open('horse_1.jpg').convert('RGB')
img = transforms(img).unsqueeze(0)  # (1, 3, 224, 224)

# Перевод модели в режим оценки
model.eval()

p = model(img).squeeze() # прогон и результат

res = p.softmax(dim=0).sort(descending=True) # прогон через softmax что бы найти вероятность самого вероятного варианта

# Прямой проход (инференс)
with torch.no_grad():
    output = model(img).squeeze()

```
