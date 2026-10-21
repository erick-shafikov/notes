# sklearn.metrics

## Классификация

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

accuracy_score(y_true, y_pred)          # 0.8
precision_score(y_true, y_pred)         # TP / (TP + FP)
recall_score(y_true, y_pred)            # TP / (TP + FN)
f1_score(y_true, y_pred)                # 2 * P * R / (P + R)
```

### Многоклассовые метрики

Параметр `average` управляет усреднением:

| значение | что делает |
|----------|------------|
| `'binary'` | только для двух классов (по умолчанию) |
| `'macro'` | среднее по классам без учёта дисбаланса |
| `'weighted'` | взвешенное по числу объектов в классе |
| `'micro'` | глобально: TP, FP, FN суммируются по всем классам |
| `None` | вернуть массив метрик — по одной на класс |

```python
f1_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='weighted')
```

### confusion_matrix

```python
cm = confusion_matrix(y_true, y_pred)
# строки — истинные классы, столбцы — предсказанные
# cm[i, j] = число объектов класса i, предсказанных как j
# диагональ = правильные предсказания

# нормированная (доля от истинных)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
```

### classification_report

```python
print(classification_report(y_true, y_pred, target_names=['cat', 'dog']))
#               precision  recall  f1-score  support
# cat               0.67    1.00      0.80        2
# dog               1.00    0.67      0.80        3
# accuracy                            0.80        5
```

---

## ROC и AUC

```python
y_scores = clf.predict_proba(X_test)[:, 1]  # вероятности положительного класса

# AUC — площадь под ROC-кривой: 0.5 (случайная) → 1.0 (идеал)
roc_auc_score(y_true, y_scores)

# точки ROC-кривой: fpr, tpr, пороги
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
```

---

## Регрессия

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse  = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)  # sklearn ≥ 0.22
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)   # 1 — идеал, 0 — предсказание средним, <0 — хуже среднего
```
