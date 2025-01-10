<!-- работа с цветами ---------------------------------------------------------------------------------------------------------------------------->

# работа с цветами

- color-contrast()
- color-mix() - для смешивания двух цветов
- color() - Позволяет задавать цветовые пространства

```html
<div data-color="red"></div>
<div data-color="green"></div>
<div data-color="blue"></div>
```

```scss
[data-color="red"] {
  background-color: color(xyz 45 20 0);
}

[data-color="green"] {
  background-color: color(xyz-d50 0.3 80 0.3);
}

[data-color="blue"] {
  background-color: color(xyz-d65 5 0 50);
}
```

- device-cmyk()
- hsl()
- hwb()
- lab()
- lch()
- oklab()
- oklch()

- rgb()

<!-- Градиенты  ---------------------------------------------------------------------------------------------------------------------------->

# градиенты

## linear-gradient()

linear-gradient() - создает линейный градиент

```scss
.simple-linear {
  //двухцветные с плавным переходом
  background: linear-gradient(blue, pink);
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
  // подсказка для перехода 10% займет blue
  background: linear-gradient(blue, 10%, pink);
}

// поменять направление
.horizontal-gradient {
  background: linear-gradient(to right, blue, pink);
  // диагональный
  background: linear-gradient(to bottom right, blue, pink);
  // использование углов
  background: linear-gradient(70deg, blue, pink);
  background: linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c);
}
```

```scss
// можно комбинировать с изображениями
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

## radial-gradient()

```scss
// расположение центра
.radial-gradient {
  background: radial-gradient(at 0% 30%, red 10px, yellow 30%, #1e90ff 50%);
}
//размер определяется расстоянием от начальной точки (центра) до ближайшей стороны блока.
.radial-ellipse-side {
  background: radial-gradient(
    ellipse closest-side,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // устанавливает размер градиента значением расстояния от начальной точки до самого дальнего угла блока.
  background: radial-gradient(
    ellipse farthest-corner at 90% 90%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // по x - 25% от контейнера, по y -75%
  background: radial-gradient(
    circle closest-side at 25% 75%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
}
```

## conic-gradient()

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

## repeating-linear-gradient

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

- repeating-conic-gradient() - лучи из центра
- repeating-radial-gradient() - круги из центра
