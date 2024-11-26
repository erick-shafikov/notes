# Макет с несколькими столбцами. Свойство column

- columns краткая запись для:
- - column-count: 1 - количество колонок
- - column-width: auto - ширина колонки, если установить, то браузер сам высчитает сколько столбцов нужно
- [column-fill: auto | balance; позволяет определить заполнение колонок - автоматическое или с равной высотой](./css-props.md/#column-fill)
- [column-gap определит расстояние между колонок](./css-props.md/#column-gap-flex-grid)
- [column-rule = column-rule-width + column-rule-style + column-rule-color позволяет определить стилизацию колонок между столбцами](./css-props.md/#column-rule-multi-column)
- - column-rule-color - цвет
- - [column-rule-style стиль разделителя](./css-props.md/#column-rule-style)
- - [column-rule-width](./css-props.md/#column-rule-width)
- [column-span позволяет растянуть элемент на несколько колонок ](./css-props.md/#column-span-multi-column)

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

- [break-after, break-before,break-inside как разрывы страниц, столбцов или регионов должны вести себя после (до) сгенерированного блока](./css-props.md/#break-after)
- [orphans - (нет в ff) Управление разрывами сколько строки минимум внизу должно оставаться](./css-props.md/#orphans)

так же работают свойства align-content к блочной оси, justify-content к встроенной оси.
Даже если контент разбит на составные элементы раскрепление по колонкам определяется свойствами column
Могут быть наслоение если в колонке есть изображение и оно шире колонки
