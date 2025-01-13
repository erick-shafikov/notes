так же работают свойства align-content к блочной оси, justify-content к встроенной оси.
Даже если контент разбит на составные элементы раскрепление по колонкам определяется свойствами column
Могут быть наслоение если в колонке есть изображение и оно шире колонки

# columns:

Устанавливает количество колонок и их ширину

Свойство позволяет разделить на столбцы текст в контейнере

```scss
 {
  /* количество */
  columns: auto;
  columns: 2;

  /* Количество и ширина */
  columns: 2 auto;
  columns: auto 12em;
  columns: auto auto;

  // разделение текста на две колонки (*)
  -moz-column-count: 2;
  column-count: 2;
  // размер промежутка между колонками (*)
  column-gap: 4rem;
  -moz-column-gap: 4rem;
  // разделитель (*)
  column-rule: 1px solid $color-grey-light-2;
  -moz-column-rule: 1px solid $color-grey-light-2;
  //позволяет растянуть элемент по ширине всех колонок
  column-span: all;
}
```

## column-count

количество колонок

## column-width

auto - ширина колонки, если установить, то браузер сам высчитает сколько столбцов нужно

## column-fill

позволяет определить заполнение колонок - автоматическое или с равной высотой

```scss
 {
  column-fill: auto; //Высота столбцов не контролируется.
  column-fill: balance; //Разделяет содержимое на равные по высоте столбцы.
}
```

# column-gap (flex, grid, multi-column)

расстояние по вертикали, определит расстояние между колонок

```scss
.column-gap {
  column-gap: auto; //1em
  column-gap: 20px;
}
```

# column-rule:

column-rule-width + column-rule-style + column-rule-color позволяет определить стилизацию колонок между столбцами

Устанавливает цвет границы между колонками = column-rule-width + column-rule-style + column-rule-color

```scss
.column-count {
  // column-count: 3;
  column-rule: solid 8px;
  column-rule: solid blue;
  column-rule: thick inset blue;
}
```

## column-rule-color

цвет колонок

```scss
.column-rule-color {
  column-rule-color: red;
  column-rule-color: rgb(192, 56, 78);
  column-rule-color: transparent;
  column-rule-color: hsla(0, 100%, 50%, 0.6);
}
```

## column-rule-style

Стиль разделителя

```scss
 {
  column-rule-style: none;
  column-rule-style: hidden;
  column-rule-style: dotted;
  column-rule-style: dashed;
  column-rule-style: solid;
  column-rule-style: double;
  column-rule-style: groove;
  column-rule-style: ridge;
  column-rule-style: inset;
  column-rule-style: outset;
}
```

## column-rule-width:

Ширина колонки

```scss
 {
  column-rule-width: thin;
  column-rule-width: medium;
  column-rule-width: thick;

  /* <length> values */
  column-rule-width: 1px;
  column-rule-width: 2.5em;
}
```

# column-span (multi-column)

```scss
.column-span {
  column-span: none;
  column-span: all;
}
```

```html
<!-- контейнер для определения колонок -->
<article>
  <!-- контент для распределения на колонки -->
  <h2>Header spanning all of the columns</h2>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
</article>
```

```scss
article {
  columns: 3;
}

h2 {
  column-span: all;
}
```

```scss
.container {
  // для элемента, который нужно вырвать из контекста и продлить по всей ширине
  column-span: all;
}
```

Внутри контейнера контент можно разделить на количество столбцов

```scss
.container {
  column-count: 3;
  // или определить размер одной колонки, браузер сам посчитает
  column-width: 200px;
  // для определения расстояния между столбцами можно использовать
  column-gap: 20px;
// для стилизации линии, которая будет разделять колонки
  column-rule: 4px dotted rgb(79, 185, 227);
  column-span
}
```

# break-after

break-before,break-inside как разрывы страниц, столбцов или регионов должны вести себя после (до) сгенерированного блока

# orphans (-ff)

Минимальное число строк, которое можно оставить внизу фрагмента перед разрывом фрагмента. Значение должно быть положительным.

```scss
 {
  orphans: 3;
}
```
