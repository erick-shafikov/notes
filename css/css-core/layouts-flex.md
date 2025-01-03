<!-- Сетки. Flex ----------------------------------------------------------------------------------------------------------------------------->

# Сетки. Flex

Позволяет управлять потоком в одном направлении пространства. Устанавливаем родителю. Сам блок не генерирует контейнеры, только дочерние элементы

```scss
 {
  display: flex;
  //  идти в ряд в блоке слева на право (по умолчанию)
  display: inline-flex;
  //  если  расставить inline элементы как flex блоки, flex – контейнер займет ширину по содержимому
}
```

При определении display === flex:

- Элементы отображаются в ряд
- позиционирование начинается от начала главной оси
- элементы не растягиваются, но могут сжиматься
- элементы растягиваются что бы занять размер побочной оси
- flex-basis: auto, flex-wrap: nowrap

Настройка контейнера:

- [позволяет настроить поток во flex контейнере](./css-props.md#flex-direction)
- [За возможность переноса во второй ряд отвечает flex-wrap по умолчанию nowrap](./css-props.md#flex-wrap)
- [flex-direction + flex-wrap = flex-flow](./css-props.md#flex-flow)
- [за минимальный размер элемента в строке отвечает flex-basis, если не задан box-sizing](./css-props.md#flex-basis)

Размеры флекс-элементов рассчитываются как в блочной модели, работает box-sizing, ужимаются по содержимому, не работает float, внешние отступы не схлопываются и не выпадают. Элементы изначально выстроятся по содержимому, некоторые будут вытянуты

!!!margin важнее выравнивания
!!!флексы ни чего не знают про форматирование

Настройка расположения элементов по осям:

- [выравнивание по главной оси justify-content](./css-props.md#justify-content-flex)
- [Выравнивание по поперечной оси align-items](./css-props.md#align-items-flex)
- - [короткая запись для align-content + justify-content = place-content](./css-props.md/#place-items)
- - [lace-items = align-items + justify-items]
- - [короткая запись для align-self + justify-self = place-self](./css-props.md/#place-self-grid-flex)

## Настройка расположения контента, при переполнении (flex-wrap: wrap):

- [распределят пространство, при переносе контента align-content](./css-props.md#align-content-flex)

если у родителя задать height: auto, то align-content не имеет влияния. У родителя нужна фиксированная высота, которая задала бы пустое пространство. При переполнении, создается новый flex-контейнер

# flex-props

Свойства flex для управления положительным и отрицательным свободным пространством. Отрицательное при переполнении

## flex-grow (flex-element)

Свойство, которое определяет как распределяет свободное пространство между элементами, при этом будет изменяться ширина элемента. Свойство начинает работать, когда есть свободное пространство
flex-grow: 1 (займет 33% от flex-контейнера если три элемента со свойством flex-grow: 1. 1 – элемент жадный, 0 - нежданный)

За растягивание flex отвечает свойство flex-grow: 0 значение по умолчанию, если flex-grow делят этот блок пропорционально в качестве пропорции. Работает для строки, которая имеет свободное пространство, при переносе и flex-grow: 1 у элемента, который остается на месте не происходит центрирование, а элемент без flex-grow: 1 будет центрироваться

## flex-shrink (flex-element)

Коэффициент сжатия. При значении nowrap, свойство начинает работать, когда на flex-элемент не остается пространства, определяет с какой скоростью готов отдавать от собственной ширины. Если это значение 0, то в ширине изменяться не будет, работает как жесткий min-width. Предел min-content

## flex-basis (flex-element)

По умолчанию auto, определяет базовую ширину(flex-direction: row) или высоту (flex-direction: column), размер flex-элемента до растягивания и сжимания, изменяется при уменьшении VP (не измениться при flex-shrink = 0), элемент не может быть больше этого размера, работает как min-width, но при flex-direction: column будет определять высоту элемента. Изначально === width, по умолчанию === max-content

```scss
// ширина будет равна 100px
.flex-element {
  flex-basis: auto;
  width: 100px;
}

// ширина будет основана на содержимом
.flex-element {
  flex-basis: auto;
}

// min-content
.flex-element {
  flex-basis: 10% | 100px; //но не ноль
  width: 100px;
}

// размер элемента не будет учитываться
.flex-element {
  flex-basis: 0;
}
```

## flex (flex-element)

flex-grow свойство задается может задаваться тремя значениями:

- flex-grow по умолчанию 0
- flex-shrink по умолчанию 1
- flex-basis по умолчанию auto
- По умолчанию – flex: 0 1 auto
- flex: auto === flex: 1 1 auto - пространство распределится равномерно, но больший займет больше места
- flex: 1 === flex: 1 1 0 - все элементы одинаковой ширины

```scss
// Запрет на расширение и сжатие flex-элемента
.box > * {
  flex: 0 0 33.3333%;
}

// все одного размер невзирая на их собственный размер
.class {
  flex: 1 1 0;
}
```

## flex-gaps (flex-container)

Позволяет настроить расстояния между flex-элементами

- [gap расстояние между элементами](./css-props.md/#gap)
- - [row-gap расстояние между элементами по главной оси](./css-props.md/#row-gap-flex-grid)
- - [column-gap расстояние по перпендикулярной оси](./css-props.md/#column-gap-flex-grid)

visibility: collapse позволяет управлять сокрытием элемента, но оставляет за ним пространство для распределение flex элементов, разница между visibility: hidden - элемент удаляется, при visibility: collapse нет

## order (flex-element)

Позволяет поменять порядок элементов, при отрицательных значениях будет находится в самом начале

Персонально распределить элементы:

- похож на align-items, только применяется к отдельным элементам, можно изменить привязку одно из контейнера
- поменять порядок элементов можно с помощью значения order. Позволяет изменить порядок элементов, значение -1 смещает влево

```css
.flex {
  display: flex;
  flex-direction: row;
  align-items: start;
}
.flex div:nth-child(2) {
  align-self: center;
  /* один из элементов (второй ребенок) */
}
/* max- и min- width, height .. применяются в конце */
.flex div:nth-child(1) {
  /* станет последним */
  order: 1;
}
.flex div:nth-child(2) {
  /* будет первым, так как отрицательный */
  order: -1;
}
```

## visibility: collapse

Позволяет оставить распорку во flex-контейнере, элемент не отображается, но учитывается в сетке

## PB. Footer

Спустить footer в самый низ

```html
<div class="card">
  <div class="content">
    <p>This card doesn't have much content.</p>
  </div>
  <footer>Card footer</footer>
</div>
```

```scss
// на контейнере
.card {
  display: flex;
  flex-direction: column;
}

.card .content {
  // на контенте
  flex: 1 1 auto;
}
```

## PB. распределение элементов с помощью margin: auto

```html
<div classname="container">
  <div></div>
  <div></div>
  <div classname="element"></div>
  <div></div>
</div>
```

```scss
.container {
  display: flex;
}

.element {
  margin-left: auto;
}
```

## PB. Откинуть контент вниз карты

```scss
.container {
  flex-direction: column;
}

.expand-content {
  flex: 1;
}
```

# BP. липкий footer

```scss
.wrapper {
  box-sizing: border-box;
  min-height: 100%;
  display: flex;
  flex-direction: column;
}
.page-header,
.page-footer {
  flex-grow: 0;
  flex-shrink: 0;
}
.page-body {
  flex-grow: 1;
}
```

# BP. хлебные крошки

```scss
.breadcrumb ul {
  display: flex;
  flex-wrap: wrap;
  list-style: none;
  margin: 0;
  padding: 0;
}

.breadcrumb li:not(:last-child)::after {
  display: inline-block;
  margin: 0 0.25rem;
  content: "→";
}
```
