```python
import torch.cuda

# доступна ли cuda - для работы с графическим процессором
is_cuda = torch.cuda.is_available()
# выбор режима работы
device = torch.device('cuda' if is_cuda else 'cpu')
t = torch.FloatTensor([-1, 0, 1, 2])
# где находится тензор
# если тензоры находятся на разных устройствах, то при действиях ошибка
# -1 => cpu
# >0 => gpu
d = t.get_device()
# определить тензор в gpu
t_gpu = torch.FloatTensor([-1, 0, 1, 2]).to('cuda')
t_gpu.cuda()
# на cpu
t_gpu.cpu()
# задание зерна на устройствах
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)  # для всех
# для отключения стах. процессов при генерации случайных чисел
# сбрасывать настройки при обучении
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

```