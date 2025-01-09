<!-- Таблицы --------------------------------------------------------------------------------------------------------------------------------->

# table-layout

позволяет управлять расположением элементов в таблице

```scss
.table-layout {
  table-layout: fixed; //не будет адаптировать
  table-layout: auto; //будет адаптировать таблицу под контент, а именно растягивать ячейки
}
```

# border-collapse (таблицы)

<!--border-collapse------------------------------------------------------------------------------------------------------------------------->

Как ведет себя рамка,по умолчанию есть расстояние между ячейками

```scss
 {
  border-collapse: collapse; //соединить границы
  border-collapse: separate; //разъединить границы таблицы
}
```

# caption-side

определяет где будет находится caption-side в таблице снизу или сверху

```scss
 {
  caption-side: top;
  caption-side: bottom; // <caption /> будет расположен внизу
}
```

# border-spacing

Расстояние между ячейками

```scss
.border-spacing {
  /* <length> */
  border-spacing: 2px;

  /* horizontal <length> | vertical <length> */
  border-spacing: 1cm 2em;
}
```

# empty-cells

Показывать или нет пустые ячейки

```scss
.empty-cells {
  empty-cells: show | hide;
}
```

зебра

```scss
tbody tr:nth-child(odd) {
  background-color: #ff33cc;
}

tbody tr:nth-child(even) {
  background-color: #e495e4;
}
```
