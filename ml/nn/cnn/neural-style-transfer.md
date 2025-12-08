```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

############################### 

# I. Подготовка изображений
# два изображения:
# которое будем изменять, загружаем
img = Image.open('img.jpg').convert('RGB')
# стиль-фильтр
img_style = Image.open('img_style.jpg').convert('RGB')

# преобразования в тензоры нужного типа
# transforms = models.VGG19_Weights.DEFAULT.transforms()
transforms = tfs_v2.Compose(
    [
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(torch.float32, scale=True),
    ]
)

# Преобразуем изображения к векторам, добавим оси так как первая ось - batch
img = transforms(img).unsqueeze(0)
img_style = transforms(img_style).unsqueeze(0)
# создаем изображения для стилизации, его будем менять
# Так как пиксели будем изменять градиентным алгоритмом, то requires_grad_
img_create = img.clone()
img_create.requires_grad_(True)


###############################

# II. Подготовка модели
# создание модели, которая вернет только результаты со слоев
class ModelStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # 5.1 создаем vgg19 без полносвязаных слоев
        # 5.2 выделяем только сверточные слои, нам не нужен полносвязный слой
        _model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mf = _model.features

        # 5.3 отключаем градиенты так как не будем обучать
        # переводим в режим эксплуатации
        self.mf.requires_grad_(False)
        self.requires_grad_(False)
        self.mf.eval()
        # 5.4 индексы слоев из которых будут формироваться для матрицы Грама
        self.idx_out = (0, 5, 10, 19, 28, 34)
        # 5.5 количество слоев
        self.num_style_layers = len(self.idx_out) - 1  # последний слой для контента

    # 5.5 модель принимает с батчем 
    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.mf):
            x = layer(x)
            if idx in self.idx_out:
                # сохраняем без батча
                outputs.append(x.squeeze(0))

        # 5.6 возвращает данные с нужных сверточных слоев
        return outputs


###############################
# III. Получение тензоров с нужных сверточных слоев
model = ModelStyle()
# пропуск через НС контентного изображения и получение тензоров с нужных слоев
outputs_img = model(img)
# пропуск через НС изображения стиля и получение тензоров с нужных слоев
outputs_img_style = model(img_style)


###############################


# IV. Дополнительные функции
# вычисление потерь по контенту
def get_content_loss(base_content, target):
    return torch.mean(torch.square(base_content - target))


# получение матриц Грама
def gram_matrix(x):
    # узнаем количества каналов
    channels = x.size(dim=0)
    # новое представление
    g = x.view(channels, -1)
    gram = torch.mm(g, g.mT) / g.size(dim=1)
    return gram


# вычисления потерь по стилю
# base_style - наборы с выбранных сверточных слоев
# gram_target - вычисленные матрица грамма для стилевого изображения
# gram_target будут вычислены один раз
def get_style_loss(base_style, gram_target):
    # веса
    style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]

    _loss = 0
    i = 0
    for base, target in zip(base_style, gram_target):
        gram_style = gram_matrix(base)
        _loss += style_weights[i] * torch.mean(torch.square(gram_style - target))
        i += 1

    return _loss


###############################
# V. Обучение

# матрицы Грама для стилизованных изображений, не включая последний
gram_matrix_style = [gram_matrix(x) for x in outputs_img_style[:model.num_style_layers]]
# Обучение
content_weight = 1
style_weight = 1000
best_loss = -1
epochs = 100

# в качестве оптимизации указываем пиксели изображения
optimizer = optim.Adam(params=[img_create], lr=0.01)
# для сохранения лучшего результата
best_img = img_create.clone()

for _e in range(epochs):
    # тензоры со сверточных слоев
    # эта копия нужна для сравнения, так как другая используется для получения матриц Грама

    # outputs_img_create - будет изменяемым объектом, который будет проходить через градиентный алгоритм
    outputs_img_create = model(img_create)  # img_create = img.clone() результат текущего состояния изображения
    # потери по контенту, берем с последнего слоя значение у результатов, так как каждый проход это сравнение между 
    # outputs_img_create - прогон копии img через НС, формируемое изображение, которое будет подвержено изменениям
    # outputs_img - результат фиксированный, срез результата прохода изображения через НС, outputs_img = model(img)
    loss_content = get_content_loss(outputs_img_create[-1], outputs_img[-1])
    # потери по стилю
    # берем все матрица Грама со всех слоев (изменяемого, стилевого)
    loss_style = get_style_loss(outputs_img_create, gram_matrix_style)
    # вычисление потерь
    loss = content_weight * loss_content + style_weight * loss_style

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ограничиваем от 0 до 1
    img_create.data.clamp_(0, 1)

    # вычисление потерь и сохранение лучшего результата
    if loss < best_loss or best_loss < 0:
        best_loss = loss
        best_img = img_create.clone()

    print(f'Iteration: {_e}, loss: {loss.item(): .4f}')

# Подготовка к выводу на экран
x = best_img.detach().squeeze()
low, hi = torch.amin(x), torch.amax(x)
x = (x - low) / (hi - low) * 255.0
x = x.permute(1, 2, 0)
x = x.numpy()
x = np.clip(x, 0, 255).astype('uint8')

image = Image.fromarray(x, 'RGB')
image.save("result.jpg")

print(best_loss)
plt.imshow(x)
plt.show()

```