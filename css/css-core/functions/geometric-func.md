<!-- геометрические фигуры  ---------------------------------------------------------------------------------------------------------------------------->

# геометрические фигуры

```scss
 {
  // круг
  clip-path: circle(50px);
  clip-path: circle(6rem at right center);
  clip-path: circle(10% at 2rem 90%);
  // для создания эллипса
  clip-path: ellipse();
  // определяет область round 50px - border радиус
  clip-path: inset(45px 50px 15px 0 round 50px);
  // определяет svg
  clip-path: path(
    "M  20  240 \
 L  20  80 L 160  80 \
 L 160  20 L 280 100 \
 L 160 180 L 160 120 \
 L  60 120 L  60 240 Z"
  );
  clip-path: polygon(
    0% 20%,
    60% 20%,
    60% 0%,
    100% 50%,
    60% 100%,
    60% 80%,
    0% 80%
  );
  // прямоугольник
  clip-path: rect(50px 70px 80% 20%);
  //произвольная прямая
  clip-path: xywh(1px 2% 3px 4em round 0 1% 2px 3em);
}
```
