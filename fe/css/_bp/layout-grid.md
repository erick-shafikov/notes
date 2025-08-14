# GRID сетка

```scss
.container {
  display: grid;
  grid-template-rows: 80vh min-content 40vw repeat(3, min-content);
  //разбиваем на 8 колонок (I)
  //первая колонка - для sidebar - 8rem
  // втора колонка в роли border - минимум 6rem максимум 1fr
  // c 3 по 7 - колонки растягиваются автоматически или занимают 14rem 1400/8
  // последняя колонка, как вторая
  grid-template-columns:
    [sidebar-start] 8rem [sidebar-end full-start] minmax(6rem, 1fr)
    [center-start] repeat(8, [col-start] minmax(min-content, 14rem) [col-end])
    [center-end]
    minmax(6rem, 1fr)
    [full-end];

  @media only screen and (max-width: $bp-large) {
    // добавляем строку сверху (II)
    grid-template-rows: 6rem 80vh min-content 40vw repeat(3, min-content);
    grid-template-columns:
        // убираем side-bar на вверх меняя сетку (II)

      [full-start] minmax(6rem, 1fr)
      [center-start] repeat(8, [col-start] minmax(min-content, 14rem) [col-end])
      [center-end]
      minmax(6rem, 1fr)
      [full-end];
  }

  @media only screen and (max-width: $bp-medium) {
    // смещаем секцию с риелторами в отдельный ряд (III)
    // что бы heading занимал весь экран нужно вычесть 6rem (высота sidebar)
    grid-template-rows: 6rem calc(100vh - 6rem);
  }
}
```

Привязка к колонке

```scss
.features {
  // занимает центральную колонку с repeat (I)
  grid-column: center-start / center-end;
}
```

Источник курс Shmidtman, там grid, проект nexter
