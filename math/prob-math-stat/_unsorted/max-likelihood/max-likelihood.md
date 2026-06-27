# Метод максимального правдоподобия

## Дискретная задача

Есть некоторое событие, которое распределяется следующим образом:

| $y_i$ | $0$ | $1$  | $2$      | $3$ |
| ----- | --- | ---- | -------- | --- |
| $p_i$ | $p$ | $2p$ | $1 - 3p$ | $0$ |

Ограничение на параметр: $p > 0$ и $1 - 3p > 0$, т.е. $0 < p < \tfrac{1}{3}$.

**Выборка:** $y_1 = 0,\ y_2 = 1,\ y_3 = 2,\ y_4 = 0$

Найти $\hat{p}_{\mathrm{ML}}$.

$$L(p) = p(y_1{=}0)\cdot p(y_2{=}1)\cdot p(y_3{=}2)\cdot p(y_4{=}0) = p \cdot 2p \cdot (1-3p) \cdot p = 2p^3(1-3p) \to \max$$

Основной приём — переход от произведения к сумме:

$$\ln L(p) = \ln 2 + 3\ln p + \ln(1-3p)$$

$$\frac{d\ln L}{dp} = \frac{3}{p} - \frac{3}{1-3p} = 0$$

$$3(1-3p) = 3p \implies 3 - 9p = 3p \implies 12p = 3$$

$$\boxed{\hat{p}_{\mathrm{ML}} = \frac{1}{4}}$$

**Проверка допустимости:** $0 < \tfrac{1}{4} < \tfrac{1}{3}$ ✓

## Непрерывные величины

**Дано:** 100 наблюдений $y_1 = 1{,}1,\ y_2 = 2{,}7,\ \ldots,\ y_{100} = 1{,}5$, где $\sum_{i=1}^{100} y_i = 200$.

Плотность (показательное распределение):

$$f(y \mid \lambda) = \begin{cases} \lambda e^{-\lambda y}, & y \geq 0 \\ 0, & y < 0 \end{cases}$$

Найти $\hat{\lambda}_{\mathrm{ML}}$.

### Функция правдоподобия

$y_i$ — независимые, поэтому:

$$L(\lambda) = \prod_{i=1}^{100} f(y_i) = \prod_{i=1}^{100} \lambda e^{-\lambda y_i} = \lambda^{100} \cdot e^{-\lambda \sum_{i=1}^{100} y_i} = \lambda^{100} e^{-200\lambda}$$

$$\ln L(\lambda) = \sum_{i=1}^{100} \bigl(\ln\lambda - \lambda y_i\bigr) = 100\ln\lambda - \lambda\sum_{i=1}^{100} y_i = 100\ln\lambda - 200\lambda$$

$$\frac{d\ln L}{d\lambda} = \frac{100}{\lambda} - 200 = 0$$

$$\boxed{\hat{\lambda}_{\mathrm{ML}} = \frac{100}{200} = \frac{1}{2}}$$
