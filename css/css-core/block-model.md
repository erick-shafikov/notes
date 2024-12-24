<!-- Блочная модель --------------------------------------------------------------------------------------------------------------->

# Блочная модель

Элемент может быть блочным или строчным. Поток - это расположение элементов в документе. Что бы выкинуть элемент из потока из контекста форматирования - float, position: absolute, корневой элемент (html)
Контексты форматирования:

- block formatting context, BFC
- inline formatting context
- flex formatting context

-[display - определяет блочность/строчность элемента](./css-props.md/#display)

- блочные боксы – прямоугольные области на странице, начинаются с новой строки, занимают всю доступную ширину, к ним применимы свойства width, height, элементы вокруг будет отодвинуты. Нужны для формирования структуры страницы. Занимает 100% ширины и высоту по содержимому. Если даже задать двум блоками идущим подряд ширину в 40% то они все равно расположатся друг под другом
- Строчные боксы – фрагменты текста span, a, strong, em, time у них нет переноса строки, ширина и высота зависят от содержимого, размеры задать нельзя за исключением элементов area и img. Не будут начинаться с новой строки, width, height недоступны, отступы не будут отодвигать другие элементы. Высота определяется по самому высокому элементу
- можно менять блочность/строчность через display

# свойства блочной модели

## Размеры:

- [box-sizing - позволяет управлять размерами контейнера](./css-props.md/#box-sizing)
- [width – ширина содержимого](./css-props.md/#width)
- [height – высота содержимого](./css-props.md/#height)
  для того что бы задать размеры отталкиваясь от минимальных и максимальных значений
- min-width, min-height, max-width, max-height – нужны для определения высоты контентных элементов, которые могут вывалиться
  max-width переопределяет width, но min-width переопределяет max-width. Свойства с учетом письма:
- - max-block-size
- - max-inline-size
- - min-block-size
- - min-inline-size

# Отступы и границы:

- padding – отступ от контента до рамки, при заливке заливается и padding и контент
- margin - внешние отступы, они могут быть автоматически добавлены к абзацам например, если задать margin ===0 то они схлопнуться
- - margin-trim (Только на ios) - позволяет обрезать margin
- [aspect-ratio позволяет настроить пропорции ширина/высота для контейнера](./css-props.md/#aspect-ratio)
- оптимизационные значения contain-intrinsic-block-size, contain-intrinsic-height, contain-intrinsic-inline-size, contain-intrinsic-size, contain-intrinsic-width
- inline-size - задает высоту или ширину блока в зависимости от написания
  Если ширина не задана, общая ширина равна доступному месту в родителе при схлопывании – суммируется margin, берется максимальный.
- visibility: visible | hidden | collapse не выкидывает элемент из дерева элементов, не меняет разметку
- z-index: number - позволяет выдвинуть элемент из контекста для позиционированного элемента (отличного от static) отрицательные значения понижают приоритет

- [определить стиль для всех 4х границ сразу](./css-props.md#border)
- - [предопределенные стили для border](./css-props.md#border-style-border-bottom-style-border-left-style-border-right-style-border-top-style)
- - [ширина границы](./css-props.md#border-width-border-bottom-width-border-left-width-border-right-width-border-top-width)
- - [цвет границы](./css-props.md/#border-left-color-border-right-color-border-top-color)
- - [сокращенная запись для одной из границ для нескольких ее свойств](./css-props.md#border-bottom-border-left-border-right-border-top)
- [border-image короткая запись для свойств](./css-props.md/#border-image)
- - [border-image-outset позволяет настроить расстояние от рамки до контента](./css-props.md/#border-image-outset)
- - [border-image-repeat растяжение картинки](./css-props.md/#border-image-repeat)
- - [border-image-slice настройка повтора картинки с помощью нарезания ее и повторения](./css-props.md/#border-image-slice)
- - [border-image-source установка источника изображения](./css-props.md/#border-image-source)
- - [border-image-width установка ширины границы](./css-props.md/#border-image-width)
- [border-radius закругление рамок](./css-props.md/#border-radius-border-bottom-left-radius-border-bottom-right-radius-border-top-left-radius-border-top-right-radius-border-top-right-radius)
- [box-shadow тени от контейнера](./css-props.md/#box-shadow)
- [box-decoration-break определяет поведение декорирования рамок, при переносе]()

border-block и border-inline свойства, которые полагаются на направление текста:

- Свойства для верхних и нижних границ: border-block-end-color, border-block-start-color, border-block-end-style, border-block-end-width, border-block-start-width
- Свойства для левой и правой: border-inline-end-color, border-inline-start-color, border-block-start-style, border-inline-end-style, border-inline-start-style, border-inline-end-width, border-inline-start-width
- Закругления: border-start-start-radius, border-start-end-radius, border-end-start-radius, border-end-end-radius

- [outline - стилизация внешней рамки, которая может наезжать на соседние элементы и не влияет на определение блочной модели состоит из:](./css-props.md/#outline)
- - [outline-color - цвет обводки](./css-props.md/#outline-color)
- - [outline-style - стиль границы обводки](./css-props.md/#outline-style)
- - [outline-width - ширина](./css-props.md/#outline-width)
- - [outline-offset - отступ границы]()

# z-index

Порядок наложения без z-index:

- Фон и границы корневого элемента.
- Дочерние блоки в нормальном потоке в порядке размещения(в HTML порядке).
- Дочерние позиционированные элементы, в порядке размещения (в HTML порядке).

float:

- Фон и границы корневого элемента
- Дочерние не позиционированные элементы в порядке появления в HTML
- Плавающие элементы
- Элементы, позиционируемые потомками, в порядке появления в HTML

# Направление письма

Блочная модель так же предусматривает направление текста

- [writing-mode](./css-props.md/#writing-mode)
- [свойство direction которое принимает два значения ltr и rtl]
- [text-orientation позволяет распределить символы в вертикальном направлении]
- [block-size позволяет задать высоту и ширину с учетом направленности письма](./css-props.md/#block-size)
- inset - позволяет определить top|bottom или right|left в зависимости от rtl
- - inset-block - позволяет определить top|bottom или right|left в зависимости от rtl более точные свойства для управление расположением:
- - - inset-block-end
- - - inset-block-start
- - inset-inline аналогично и inset-block только представляет горизонтальную ориентацию
- - - inset-inline-end
- - - inset-inline-start
- [text-combine-upright учет чисел при написании в иероглифах all - все числа будут упакованы в размер одного символа]

# Вытекание за контейнер

Возникает, при том условии, когда размер одного или группы элементов в сумме больше размера контейнера. одно из свойств блочной модели регулируется с помощью свойства overflow:

- [overflow - если контент больше чем контейнер](./css-props.md#overflow)
- - свойство в зависимости от направленности письма:
- - - overflow-block
- - - overflow-inline
- [overflow-x горизонтальный](./css-props.md#overflow)
- [overflow-y вертикальный скролл](./css-props.md#overflow)

# переполнение контента и скролл

Поведение при скролле:

- scroll-behavior: auto | smooth для поведения прокрутки
- (нет в Safari) scrollbar-width auto | thin | none;
- (нет в Safari)[scrollbar-color цвет scrollbar ](./css-props.md/#scrollbar-color)
- (нет в Safari) scrollbar-gutter: auto | stable | oth-edges - ;
- (нет в Safari) overflow-anchor: auto | none определяет поведения прокрутки, при добавлении элементов

Сделать скролл дискретным (при прокрутке привязывался к позиции)

- [scroll-snap-type как строго привязывается прокрутка ](./css-props.md/#scroll-snap-type)
- scroll-snap-align: center | start | end позволяет при прокрутки фиксировать позицию элемента
- scroll-margin: px позволяет прокрутить в определенное место элемента с определенным margin, является сокращенной записью для scroll-margin-right + scroll-margin-bottom + scroll-margin-left, при нуле поместит элемент в середину
- scroll-margin-inline = scroll-margin-inline-start + scroll-margin-inline-end
- scroll-margin-block = scroll-margin-block-start + scroll-margin-block-end
- scroll-padding: px позволяет прокрутить в определенное место при scroll-snap, коротка запись для группы scroll-padding-bottom + scroll-padding-left + scroll-padding-top + scroll-padding-right
- scroll-padding-inline = scroll-padding-inline-start + scroll-padding-inline-end
- scroll-padding-block = scroll-padding-block-start + scroll-padding-block-end

свойства scroll-padding- и scroll-margin- могут помочь в ситуации когда заголовок остается в фиксированном месте

- scroll-snap-stop: normal | always придает дискретность к прокрутке
- overscroll-behavior: auto | contain | none - поведение при достижении конца скролла шорткат для:
- - overscroll-behavior-x
- - overscroll-behavior-y
- - overscroll-behavior-block, overscroll-behavior-inline для поведения с учетом направленности текста

```html
<article class="scroller">
  <section>
    <h2>Section one</h2>
  </section>
  <section>
    <h2>Section two</h2>
  </section>
  <section>
    <h2>Section three</h2>
  </section>
</article>
```

```scss
.scroller {
  height: 300px;
  overflow-y: scroll;
  scroll-snap-type: y mandatory;
}

.scroller section {
  scroll-snap-align: start;
}
```

::-webkit-scrollbar - псевдоэлементы группы scrollbar:

- ::-webkit-scrollbar-button
- ::-webkit-scrollbar:horizontal{}
- ::-webkit-scrollbar-thumb
- ::-webkit-scrollbar-track
- ::-webkit-scrollbar-track-piece
- ::-webkit-scrollbar:vertical{}
- ::-webkit-scrollbar-corner
- ::-webkit-resizer

## BP. Центрирование с помощью блочной модели (margin)

```css
div {
  width: 200px;
  margin: 0 auto;
}
```

width: auto – отдаст под контент, ту часть, которая останется от padding и margin
width: 100% - займет весь контейнер, но если сумма содержимого и рамок больше размера контейнера, то содержимое выпадет за переделы контейнера

## BP. скрыть элемент оставив его доступным

```scss
 {
  visibility: hidden;
  width: 0px;
  height: 0px;
}
```
