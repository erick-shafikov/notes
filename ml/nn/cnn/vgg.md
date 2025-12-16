# VGG-16

Обученная на 1000 классификации
На вход подается 3х канальное изображение разрешением 224x224

Структура:

- Вход 3х224х224
- Слой 1:
  -- Conv2d(64,) # размер изображения 224x224
  -- Relu()
  -- Conv2d(64) # размер изображения 224x224
  -- Relu()
  -- MaxPolling(2)
- Слой 2:
  -- Conv2d(128) # размер изображения 112x112
  -- Relu()
  -- Conv2d(128) # размер изображения 112x112
  -- Relu()
  -- MaxPolling()
- Слой 3:
  -- Conv2d(256) # размер изображения 56x56
  -- Relu()
  -- Conv2d(256) # размер изображения 56x56
  -- Relu()
  -- MaxPolling(2)
- Слой 4:
  -- Conv2d(512) # размер изображения 28x28
  -- Relu()
  -- Conv2d(512) # размер изображения 28x28
  -- Relu()
  -- MaxPolling(2)
- Слой 5:
  -- Conv2d(512, 14, 14) # размер изображения 14x14
  -- Relu()
  -- Conv2d(512, 14, 14) # размер изображения 14x14
  -- Relu()
  -- MaxPolling(2)
- Слой 6:
  -- Conv2d(512, 7, 7) # размер изображения 7x7
  -- Relu()
  -- Conv2d(512, 7, 7) # размер изображения 7x7
  -- Relu()
  -- MaxPolling()
- Слой 7:
  --Linear(4096) + Relu()
  --Linear(4096) + Relu()
  -- Softmax(1000)

- везде ядра 3х3
- функция активации Relu

Способ применения

```python
from PIL import Image
import torchvision.transforms as tfs
from torchvision import models

# обращение к предобученным весам
vgg_weights = models.VGG16_Weights.DEFAULT
vgg_weights = models.VGG16_Weights.IMAGENET1K_V1 # то же самое, что и models.VGG16_Weights.DEFAULT

categories = vgg_weights.meta['categories'] # ссылка на категории
transforms = vgg_weights.transforms() # ссылка на категории
# модель vgg16
model = models.vgg16(weights='DEFAULT')
model = model.features  # ссылка на сверточные
model = model.classifier  # ссылка на полносвязные
models.vgg16(weights=vgg_weights) # другой вариант применения
img = Image.open('img_224.jpg').convert('RGB')
img = tfs.ToTensor()(img)

model.eval()
p = model(img.unsqueeze(0)).squeeze()
res = p.softmax(dim=0).sort(descending=True)
# vgg_weights содержит обученные весовые коэффициенты сети VGG-16 для классификации полноцветных изображений

# объекты трансформации
vgg_weights.transforms()
```
