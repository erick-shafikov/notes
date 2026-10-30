# Многоклассовая классификация

## 1. Микроусреднение precision и recall

```python
# Условие:
# Для заданных массивов TP,TN,FP,FN по нескольким классам вычислить precision и recall с
# микроусреднением: сначала объединить количества по классам, затем применить формулы метрик.
# Сохранить precision,recall; не печатать. Исходные массивы — в заготовке/решении.

import numpy as np

TP = np.array([45, 37, 51, 47])
TN = np.array([36, 37, 29, 28])
FP = np.array([18, 21, 15, 17])
FN = np.array([8, 11, 9, 5])

precision = TP.mean() / (TP.mean() + FP.mean())
recall = TP.mean() / (TP.mean() + FN.mean())
```

## 2. Макроусреднение precision и recall

```python
# Условие:
# Для заданных массивов TP,TN,FP,FN вычислить precision и recall с макроусреднением: рассчитать
# метрику отдельно для каждого класса, затем взять среднее. Сохранить precision,recall; ничего
# не печатать.

import numpy as np

TP = [57, 48, 60, 55]
TN = [32, 35, 28, 41]
FP = [13, 15, 12, 11]
FN = [7, 12, 11, 8]
precision = np.array([TP[i] / (TP[i] + FP[i]) for i in range(4)]).mean()
recall = np.array([TP[i] / (TP[i] + FN[i]) for i in range(4)]).mean()
```
