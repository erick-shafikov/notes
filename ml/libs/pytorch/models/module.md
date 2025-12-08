# Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    # при пропуске через слои
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


model2 = Model()

model = Model()

# ПРИВЕДЕНИЕ ПАРАМЕТРОВ К ТИПАМ
# приведет все параметры к bfloat16 
model.bfloat16()
model.double()
model.float()
model.half()
###################################

# ВЫПОЛНЕНИЕ НА УСТРОЙСТВЕ
# где выполняется
model.cpu()
model.cuda()
model.ipu()
model.mtia()
model.xpu()
###################################

# ИТЕРАЦИЯ ПО МОДУЛЯМ
# вернет итератор для модулей
model.buffers()
# вернет итератор для модулей Iterator[Module]
model.children()
model.compile()
# итератор по модулям
model.modules()
# итератор по модулям
model.named_buffers()
# возвращает итератор по непосредственным дочерним модулям, возвращая как имя модуля, так и сам модуль
model.named_children()
# возвращает итератор по модулям
model.named_modules()
###################################

# ХУКИ
# хук дял backward
model.register_backward_hook(lambda x, y, z: x)
# добавит функцию-хук на forward
model.register_forward_hook(lambda x, y, z: x)
# добавит функцию-хук на forward
model.register_forward_pre_hook(lambda x, y: x)
# добавит функцию-хук на backward посчитанного для всего модуля
model.register_full_backward_hook(lambda x, y, z: x)
model.register_full_backward_pre_hook(lambda x, y: x)
model.register_load_state_dict_post_hook(lambda x: x)
model.register_load_state_dict_pre_hook(lambda module, state_dict, prefix, local_metadata: None)
model.register_state_dict_pre_hook(lambda module, prefix, keep_vars: None)
###################################

# РАБОТА С ГРАДИЕНТАМИ
# сброс градиентов
model.zero_grad()
# перевод в обучение
model.train()
# выполнение
model.eval()
# включение отключение градиентного спуска
model.requires_grad_()
###################################

# РАЗНОЕ
# добавить модель
model.add_module('model_name', model2)
# добавить модели рекурсивно, принимает функцию
model.apply(lambda x: x)

# доп информация
model.extra_repr()
# выполнить
model.forward(torch.tensor([1, 2, 3]))
# вернет состояние
model.get_extra_state()
# параметры
model.get_parameter('model2')
# вернет вложенные
model.get_submodule('model2')
# загрузить модель в виде словаря
model.load_state_dict({'dict': 'some_data'}, strict=True, assign=False)
# итератор по параметрам модуля (имя и значение)
model.named_parameters()
# итератор по параметрам модуля только значение
model.parameters()
# добавит буфер в модуль
model.register_buffer('name', torch.tensor([1, 2, 3]))
# выполнение Alias for add_module()
model.register_module('model2', model2)
model.register_parameter('model2')
# дополнительные данные для state_dict
model.set_extra_state({'some': 'thing'})
# выполнение
model.set_submodule(nn.Linear(1, 1))
# выполнение
model.share_memory()
# множественное преобразование модели
model.to()
# тип
print(model.type)

```