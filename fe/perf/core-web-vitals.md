# Метрики страницы

## FCP

first contentful paint (первая отрисовка) выделять критический css (главное метрика)

## LCP

- largest contentful paint - отрисовка самого большого элемента. Равен FCP без картинок и без критических стилей. Начинается в момент загрузки страницы, парсится страница ищется кандидат на самый большой элемент. Заканчивается подсчет после начала взаимодействия с сайтом

Показатели:

- Время: хорошо до 2.5 с. плохо - 4с.
- факторы влияния:
- - время отклика сервера
- - код, блокирующий рендеринг,шрифты
- - фоновые картинки
- улучшения:
- - выделять height и width для картинки, нельзя использовать Opacity:0, размер должен быть менее 100% энтропия больше 0.05
- - Убивают видео и фреймы. Можно вставить прямоугольник или картинку, но появляется потенциальная проблема оптимизации
- - Если проблема в картинке убрать lazy и добавить в preload или заменить на заголовок.
- - Если текст - добавить шрифт в preload.
- - скрипты в конец body
- - важные стили в head
- - font-display: swap

## FID (deprecated)

время до того как элемент становится активным. Разбиение больших js задач на более мелкие. Или использовать css вместо js

- Время: хорошо - 100мс, плохо - 400
- Факторы влияния: объем js

## TTI

time to interactive

## CLS

Cumulative layout shift (сдвиг от первоначального) - замеряется за 5 сек. продлевается по 1 с. после каждого сдвига, формула:

layout shift score = impact fraction x distance fraction

- impact fraction - сколько элементов было сдвинуто (измеряется в % vh)
  impact fraction = view hright - element appears size / view height
- distance fraction - на сколько пикселей
  df = height / weight of banner / view height
- CLS = if x df

Параметры:

- время: хорошо - 0.1 с, плохо = 0.25
- Факторы влияния:
- - размеры картинок
- - загрузка банеров
- - резервирование места
- - шрифты-невидимки
- методы по улучшению:
- - правильно выделять критический css
- - анимировать в CSS только функцией transform: translate()
- - учитывать prefer-reduced-motion

## TBT

total blocking time – время, которое нужно для обработки первого действия пользователя

## TTFB

Time to First Byte

- размещать контент как можно ближе к пользователям географически
- кешировать контент для быстрого ответа при повторных запросах

## SI

speed index

!!! Браузер автоматически сокращает любое количество пробелов до одного
!!! больше 4гб не выделяется на вкладку

# INP

Interaction to Next Paint - Взаимодействие со следующей отрисовкой, ощущение что сайт завис. Замена FID. Считает по взаимодействию с сайтом и блокировкой потока, измеряется в мс, выбирается самое долгое за 50 взаимодействий

Факторы влияющие:

- время - хорошо до 200 мс, плохо больше 500 мс
- Оптимизация длительных задач:
- - оптимальное использование библиотек
- - dev-tools coverage tool
- - code-splitting
- Минимизация крупных обновлений рендеринга
- - минимизация операций в DOM
- - css для скрытого
- - добавление loader для отображения обратной связи
- - разбиение задач с помощью requestIdleCallback

# js сканирование аналитики

```js
import { getCLS, getFID, getLCP } from "web-vitals";

function sendToAnalytics(metric) {
  const body = JSON.stringify(metric);
  // Use `navigator.sendBeacon()` if available, falling back to `fetch()`.
  (navigator.sendBeacon && navigator.sendBeacon("/analytics", body)) ||
    fetch("/analytics", { body, method: "POST", keepalive: true });
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getLCP(sendToAnalytics);
```

# дополнительные сведения

Базы данных от google по user experience:

- RUM — Real User Monitoring
- CrUX - Chrome User Experience - разница по работе с ifarme
