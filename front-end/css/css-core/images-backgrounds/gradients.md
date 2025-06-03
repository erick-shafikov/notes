<!-- Градиенты ------------------------------------------------------------------------------------------------------------------------------->

Градиенты могут быть использованы, там где используются изображения.

# linear-gradient()

linear-gradient() - создает линейный градиент

## направление

```scss
.simple-linear {
  //двухцветные с плавным переходом, по умолчанию сверху вниз
  background: linear-gradient(blue, pink);
  // поменять на слева направо
  background: linear-gradient(to right, blue, pink);
  // диагональный
  background: linear-gradient(to bottom right, blue, pink);
  //под определенным углом
  background: linear-gradient(70deg, blue, pink);
  background: linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c);
}
```

## количество нескольких цветов

Расположение точек на полу пути

```scss
.simple-linear {
  background: linear-gradient(#e66465, #9198e5);
  // по умолчанию будут равно распределены
  background: linear-gradient(red, yellow, blue, orange);
  //распределение в неравных пропорциях
  background: linear-gradient(to left, lime 28px, red 77%, cyan);
  //конкретные промежутки
  background: linear-gradient(
    to left,
    lime 25%,
    red 25% 50%,
    cyan 50% 75%,
    yellow 75%
  );
  background: linear-gradient(to left, #333, #333 50%, #eee 75%, #333 75%);
  // для резкого перехода проценты в сумме должны быть равны 100
  background: linear-gradient(to bottom left, cyan 50%, palegoldenrod 50%);
  // подсказка для перехода 10% займет blue (blue будет меньше) одна точка показывает переход
  background: linear-gradient(blue, 10%, pink);
  //для точного расположения
  background: linear-gradient(
    to left,
    lime 20%,
    red 30% 45%,
    cyan 55% 70%,
    yellow 80%
  );
  //вариант с полосками
  background: linear-gradient(
    to left,
    lime 25%,
    red 25% 50%,
    cyan 50% 75%,
    yellow 75%
  );
}
```

```scss
// можно комбинировать с изображениями, градиенты - прозрачны
.layered-image {
  background: linear-gradient(to right, transparent, mistyrose),
    url("critters.png");
}
```

Наслоение нескольких друг на друга

```scss
 {
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

# radial-gradient()

## позиция центра

```scss
// расположение центра по умолчанию в центре - красный, по краям синий
.simple-radial {
  background: radial-gradient(red, blue);
}
// красный  на 10px от центра, желтый до 30%, с 50% - синий
.radial-gradient {
  background: radial-gradient(red 10px, yellow 30%, #1e90ff 50%);
}
// смещение центра
// центр на 0 по x и 30 от верха
.radial-gradient {
  background: radial-gradient(at 0% 30%, red 10px, yellow 30%, #1e90ff 50%);
}
```

## размер

circle и ellipse будут по разному вести себя

```scss
//градиент имеет такую форму, что заканчивается у ближайшей к центру границы элемента
//градиент растянет себя так, что бы коснуться ближайшей стороны
.radial-ellipse-side {
  background: radial-gradient(
    ellipse closest-side,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // градиент заканчивается у дальней от центра границы элемента
  //градиент растянет себя так, что бы коснуться дальней стороны стороны
  background: radial-gradient(
    ellipse farthest-corner at 90% 90%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // форма градиента подбирается таким образом, чтобы его край проходил через ближайший к центру угол
  //градиент растянет себя так, что бы коснуться ближайший угол
  background: radial-gradient(
    /*по x - 25% от контейнера, по y -75%*/ circle closest-side at 25% 75%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );

  //farthest-corner край градиента будет проходить через дальний от центра угол.
  //градиент растянет себя так, что бы коснуться дальний угол
  background: radial-gradient(
    circle farthest-corner at 25% 75%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
}
```

## Наложение

```scss
.stacked-radial {
  background: radial-gradient(
      circle at 50% 0,
      rgba(255, 0, 0, 0.5),
      rgba(255, 0, 0, 0) 70.71%
    ), radial-gradient(
      circle at 6.7% 75%,
      rgba(0, 0, 255, 0.5),
      rgba(0, 0, 255, 0) 70.71%
    ),
    radial-gradient(
        circle at 93.3% 75%,
        rgba(0, 255, 0, 0.5),
        rgba(0, 255, 0, 0) 70.71%
      ) beige;
  border-radius: 50%;
}
```

# conic-gradient()

conic-gradient() - создает круговой градиент

```scss
.conic-gradient {
  // смещение центра
  background: conic-gradient(at 0% 30%, red 10%, yellow 30%, #1e90ff 50%);
  // поворот поворота
  background: conic-gradient(
    from 45deg,
    red,
    orange,
    yellow,
    green,
    blue,
    purple
  );
}
```

# repeating-linear-gradient

repeating-linear-gradient() - линии

```scss
.repeating-linear {
  background: repeating-linear-gradient(
    -45deg,
    red,
    red 5px,
    blue 5px,
    blue 10px
  );
}
```

# repeating-conic-gradient()

лучи из центра

# repeating-radial-gradient() - круги из центра

- при создании должно быть указано как минимум два цвета

# PBs:

## BP. Клетчатый градиент

```scss
.plaid-gradient {
  background: repeating-linear-gradient(
      90deg,
      transparent,
      transparent 50px,
      rgba(255, 127, 0, 0.25) 50px,
      rgba(255, 127, 0, 0.25) 56px,
      transparent 56px,
      transparent 63px,
      rgba(255, 127, 0, 0.25) 63px,
      rgba(255, 127, 0, 0.25) 69px,
      transparent 69px,
      transparent 116px,
      rgba(255, 206, 0, 0.25) 116px,
      rgba(255, 206, 0, 0.25) 166px
    ), repeating-linear-gradient(
      0deg,
      transparent,
      transparent 50px,
      rgba(255, 127, 0, 0.25) 50px,
      rgba(255, 127, 0, 0.25) 56px,
      transparent 56px,
      transparent 63px,
      rgba(255, 127, 0, 0.25) 63px,
      rgba(255, 127, 0, 0.25) 69px,
      transparent 69px,
      transparent 116px,
      rgba(255, 206, 0, 0.25) 116px,
      rgba(255, 206, 0, 0.25) 166px
    ), repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 5px,
      rgba(143, 77, 63, 0.25) 5px,
      rgba(143, 77, 63, 0.25) 10px
    ), repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(
          143,
          77,
          63,
          0.25
        ) 5px, rgba(143, 77, 63, 0.25) 10px);

  background: repeating-linear-gradient(
      90deg,
      transparent 0 50px,
      rgba(255, 127, 0, 0.25) 50px 56px,
      transparent 56px 63px,
      rgba(255, 127, 0, 0.25) 63px 69px,
      transparent 69px 116px,
      rgba(255, 206, 0, 0.25) 116px 166px
    ), repeating-linear-gradient(
      0deg,
      transparent 0 50px,
      rgba(255, 127, 0, 0.25) 50px 56px,
      transparent 56px 63px,
      rgba(255, 127, 0, 0.25) 63px 69px,
      transparent 69px 116px,
      rgba(255, 206, 0, 0.25) 116px 166px
    ), repeating-linear-gradient(
      -45deg,
      transparent 0 5px,
      rgba(143, 77, 63, 0.25) 5px 10px
    ), repeating-linear-gradient(45deg, transparent 0 5px, rgba(
          143,
          77,
          63,
          0.25
        ) 5px 10px);
}
```

# BP. Повторяющиеся круговые градиенты

```scss
.repeating-radial {
  background: repeating-radial-gradient(
    black,
    black 5px,
    white 5px,
    white 10px
  );
}
```
