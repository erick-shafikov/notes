<!-- Сетки. Flex ----------------------------------------------------------------------------------------------------------------------------->

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

# flex-direction

позволяет настроить поток во flex контейнере

```scss
 {
  flex-direction: row; // справа на лево, то есть блоки будут идти справа на лево
  flex-direction: column; // сверху вниз, как div-ы
  flex-direction: row-reverse; // снизу вверх
  flex-direction: column-reverse;
  // общие значения
  flex-direction: inherit;
  flex-direction: initial;
  flex-direction: revert;
  flex-direction: revert-layer;
  flex-direction: unset;
}
```

# flex-wrap

За возможность переноса во второй ряд отвечает flex-wrap по умолчанию nowrap

```scss
 {
  flex-wrap: wrap; //
}
```

## flex-flow

    позволяет задать в одной строчке задать flex-direction + flex-wrap

```scss
 {
  flex-flow: row wrap;
}
```

# Выравнивание

## justify-content (flex, grid)

выравнивание по главной оси justify-content

```scss
.flex {
  justify-content: center; // Выравнивание элементов по центру
  justify-content: start; // Выравнивание элементов в начале в отличие от flex-start отчет идет от направления письма
  justify-content: end; // Выравнивание элементов в конце
  justify-content: flex-start; // Выравнивание флекс-элементов с начала
  justify-content: flex-end; // Выравнивание флекс-элементов с конца
  justify-content: left; // Выравнивание элементов по левому краю
  justify-content: right; // Выравнивание элементов по правому краю

  // Выравнивание относительно осевой линии
  justify-content: baseline;
  justify-content: first baseline;
  justify-content: last baseline;

  // Распределённое выравнивание
  justify-content: space-between; // Равномерно распределяет все элементы по ширине flex-блока. Первый элемент вначале, последний в конце
  justify-content: space-around; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют полноразмерное пространство с обоих концов
  justify-content: space-evenly; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют равное пространство вокруг
  justify-content: stretch; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют "авто-размер", чтобы соответствовать контейнеру
  // Выравнивание при переполнении
  justify-content: safe center;
  justify-content: unsafe center;
}
```

## align-items (flex, grid)

Выравнивание по поперечной оси

```scss
.flex {
  /* высота по умолчанию  */
  align-items: stretch;
  /*по верхнему краю (к верху) */
  align-items: start;
  /*по нижнему краю (к низу) */
  align-items: end;
  /*отцентрирует */
  align-items: center;
  /* выравнивает текст внутри элементов */
  align-items: baseline;
}
```

## place-items (grid, flex)

короткая запись place-items = align-items + justify-items

```scss
 {
  place-items: end center;
}
```

## place-self (grid, flex)

place-self = align-self + justify-self

```scss
 {
  place-self: stretch center;
}
```

# Настройка расположения контента, при переполнении (flex-wrap: wrap):

## align-content (flex)

распределят пространство, при переносе контента align-content

```scss
.align-content {
  align-content: flex-start;
  align-content: flex-end;
  align-content: center;
  align-content: space-between;
  align-content: space-around;
  align-content: stretch;
}
```

если у родителя задать height: auto, то align-content не имеет влияния. У родителя нужна фиксированная высота, которая задала бы пустое пространство. При переполнении, создается новый flex-контейнер

# flex:

flex-grow свойство задается может задаваться тремя значениями:

- flex-grow по умолчанию 0
- flex-shrink по умолчанию 1
- flex-basis по умолчанию auto
- По умолчанию – flex: 0 1 auto
- flex: auto === flex: 1 1 auto - пространство распределится равномерно, но больший займет больше места
- flex: 1 === flex: 1 1 0 - все элементы одинаковой ширины

```scss
// шорткат для flex-grow + flex-shrink + flex-basis
 {
  flex: 1 1 200px;
}
```

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

Свойства flex для управления положительным и отрицательным свободным пространством. Отрицательное при переполнении

## flex-grow (flex-element)

Свойство, которое определяет как распределяет свободное пространство между элементами, при этом будет изменяться ширина элемента. Свойство начинает работать, когда есть свободное пространство
flex-grow: 1 (займет 33% от flex-контейнера если три элемента со свойством flex-grow: 1. 1 – элемент жадный, 0 - нежданный)

За растягивание flex отвечает свойство flex-grow: 0 значение по умолчанию, если flex-grow делят этот блок пропорционально в качестве пропорции. Работает для строки, которая имеет свободное пространство, при переносе и flex-grow: 1 у элемента, который остается на месте не происходит центрирование, а элемент без flex-grow: 1 будет центрироваться

## flex-shrink (flex-element)

Коэффициент сжатия. При значении nowrap, свойство начинает работать, когда на flex-элемент не остается пространства, определяет с какой скоростью готов отдавать от собственной ширины. Если это значение 0, то в ширине изменяться не будет, работает как жесткий min-width. Предел min-content

## flex-basis (flex-element)

Устанавливает минимальное значение размера flex-элемента, если не задан box-sizing? если оно не установлено блочной моделью.

Разница между flex-basis 0 и 0% в том что, во втором случает элемент ужмется до своих минимальных размеров внутреннего контента

```scss
 {
  flex-basis: auto; // значение по умолчанию
  flex-basis: fill;
  flex-basis: max-content;
  flex-basis: min-content;
  flex-basis: fit-content;
  flex-basis: content; // определяет размер на основе содержимого
  flex-basis: 0; // определяет пропорционально с другими элементами
  flex-basis: 100px; // если в px то определяет минимальный размер контейнера
}
```

Размеры флекс-элементов рассчитываются как в блочной модели, работает box-sizing, ужимаются по содержимому, не работает float, внешние отступы не схлопываются и не выпадают. Элементы изначально выстроятся по содержимому, некоторые будут вытянуты

!!!margin важнее выравнивания
!!!флексы ни чего не знают про форматирование

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

# gap:

Позволяет настроить расстояния между flex-элементами

сокращенная запись gap = row-gap + column-gap

gap расстояние между элементами

```scss
 {
  gap: 10px 20px;
}
```

### row-gap (flex, grid)

расстояние по горизонтали

```scss
 {
  row-gap: 20px;
}
```

### column-gap

расстояние по перпендикулярной оси

visibility: collapse позволяет управлять сокрытием элемента, но оставляет за ним пространство для распределение flex элементов, разница между visibility: hidden - элемент удаляется, при visibility: collapse нет

# order (flex-element)

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

# visibility: collapse

Позволяет оставить распорку во flex-контейнере, элемент не отображается, но учитывается в сетке

# BPs:

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

## BP. липкий footer

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

## BP. хлебные крошки

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
