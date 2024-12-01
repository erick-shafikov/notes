Советы по улучшению работы с изображениями:

- использование правильных форматов
- компрессия изображений
- lazy loading
- адаптивные изображения
- использование cdn

# Графика и изображения

Использования изображения как фон:

- [background - использование свойства background для заливки заднего фона](./css-props#background)
- - [background-attachment - режимы прокрутка заднего фона при скролле](./css-props.md/#background-attachment)
- - [background-blend-mode - определяет как будут смешиваться наслаиваемые цвета и изображения](./css-props.md/#background-blend-mode)
- - [webkit-background-clip - обрезка изображения](./css-props.md#webkit-background-clip)
- - [background-image - добавление градиента или ссылку на изображение](./css-props.md#background-image)
- - [background-origin - расположение изображения в контейнере](./css-props.md#background-origin)
- - [background-position - изменить координаты расположения в контейнере](./css-props.md#background-position)
- - - [вертикальное и горизонтально расположение изображения](./css-props.md#background-position-x-и-background-position-y)
- - [background-repeat - повтор изображения](./css-props.md#background-repeat)
- - [background-size - растягивание и размер изображения](./css-props.md#background-size)

clip-path:

- [использование свойства clip-path](./css-props#clip-path)
- - [clip-rule: nonzero | evenodd настрой выбора пикселей для вычета]

mask - краткая запись следующих свойств нужная для маскирования изображения:

- - [mask-clip определяет область применения маски](./css-props.md#mask-clip)
- - [mask-image определяет url маски](./css-props.md/#mask-image)
- - [mask-mode: alpha | luminance | match-source]
- - [mask-origin определяет расположение начала](./css-props.md/#mask-origin)
- - [mask-position: 25% 75% позиция top/left ]
- - [mask-repeat степень повторения](./css-props.md/#mask-repeat)
- - [mask-size размер](./css-props.md/#mask-size)
- - [mask-type: luminance | alpha тип маски ]

mask-border (экспериментальное) краткая запись следующих свойств позволяет создать маску для границ:

- - mask-border-mode: luminance | alpha использование яркости или альфа-значения в качестве маски
- - mask-border-outset: 7px 12px 14px 5px; отступы
- - mask-border-repeat: stretch | repeat | round | space применение
- - mask-border-slice: 7 12 14 5
- - mask-border-source: url(image.jpg); источник
- - mask-border-width: 5% 2em 10% auto; размеры

Фильтры:

- [использование свойства filter](./css-props#filter)
- [функции для свойства filter](./functions.md/#filter-функции)
- [использование фильтра, который будет применяться к контенту, который находится поверх background-color или image](./css-props.md/#backdrop-filter)
- [для упаковки тега img в контейнер](./css-props.md#object-fit)
- [расположение изображения в контейнере object-position](./css-props.md/#object-position)
- [тени](./css-props.md#box-shadow)
- Текст, который залит фоном:
- - [обрезка фона под текст](./css-props.md#webkit-background-clip)
- - [заливка](./css-props.md#webkit-text-fill-color)

image-свойства

- image-orientation: none | from-image позволяет клиенту автоматически перевернуть изображение
- image-rendering: auto | crisp-edges | pixelated позволяет сгладить края при возникновении пикселей в изображении
- image-resolution (экспериментальное) управление качеством

# Градиенты

Градиенты могут быть использованы, там где используются изображения.

[gradient() - что бы создать функцию ](./functions.md#градиенты)

- при создании должно быть указано как минимум два цвета

## BP

### BP. Дефолтный стиль для img

!!! При работе с изображениями

```scss
 {
  max-width: 100%;
  height: auto;
} //запретить вытекание за родительский контейнер
```

### BP. Центрирование изображения

```css
img {
  display: block;
  margin: 0 auto;
}
```
