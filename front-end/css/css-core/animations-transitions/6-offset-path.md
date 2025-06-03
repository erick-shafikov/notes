# Анимация движения по пути offset-path

Позволяет анимировать объект который следует по пути

## offset:

offset = offset-anchor + offset-distance + offset-path + offset-position + offset-rotate

позволяет определить траекторию

```scss
 {
  offset: 10px 30px;

  /* Offset path */
  offset: ray(45deg closest-side);
  offset: path("M 100 100 L 300 100 L 200 300 z");
  offset: url(arc.svg);

  /* Offset path with distance and/or rotation */
  offset: url(circle.svg) 100px;
  offset: url(circle.svg) 40%;
  offset: url(circle.svg) 30deg;
  offset: url(circle.svg) 50px 20deg;

  /* Including offset anchor */
  offset: ray(45deg closest-side) / 40px 20px;
  offset: url(arc.svg) 2cm / 0.5cm 3cm;
  offset: url(arc.svg) 30deg / 50px 100px;
}
```

### offset-anchor

Позволяет определить где будет находится элемент относительно прямой при движение по линии

```scss
 {
  offset-anchor: top;
  offset-anchor: bottom;
  offset-anchor: left;
  offset-anchor: right;
  offset-anchor: center;
  offset-anchor: auto;

  /* <percentage> values */
  offset-anchor: 25% 75%;

  /* <length> values */
  offset-anchor: 0 0;
  offset-anchor: 1cm 2cm;
  offset-anchor: 10ch 8em;

  /* Edge offsets values */
  offset-anchor: bottom 10px right 20px;
  offset-anchor: right 3em bottom 10px;
}
```

### offset-distance

px | % стартовая точка где будет находится элемент

### offset-path:

offset-distance + offset-rotate + and offset-anchor

Позволяет задать путь движения

```scss
 {
  offset-path: ray(45deg closest-side contain);
  offset-path: ray(contain 150deg at center center);
  offset-path: ray(45deg);

  /* URL */
  offset-path: url(#myCircle);

  /* Basic shape */
  offset-path: circle(50% at 25% 25%);
  offset-path: ellipse(50% 50% at 25% 25%);
  offset-path: inset(50% 50% 50% 50%);
  offset-path: polygon(30% 0%, 70% 0%, 100% 50%, 30% 100%, 0% 70%, 0% 30%);
  offset-path: path(
    "M 0,200 Q 200,200 260,80 Q 290,20 400,0 Q 300,100 400,200"
  );
  offset-path: rect(5px 5px 160px 145px round 20%);
  offset-path: xywh(0 5px 100% 75% round 15% 0);

  /* Coordinate box */
  offset-path: content-box;
  offset-path: padding-box;
  offset-path: border-box;
  offset-path: fill-box;
  offset-path: stroke-box;
  offset-path: view-box;
}
```

## offset-position

смещение относительно начала

## offset-rotate

вращение элемента относительно себя

## ray()

Отклонение от оси при создании анимации по clip-path

```scss
/* all parameters specified */
offset-path: ray(50deg closest-corner contain at 100px 20px);

/* two parameters specified, order does not matter */
offset-path: ray(contain 200deg);

/* only one parameter specified */
offset-path: ray(45deg);
```

```scss
#motion-demo {
  offset-path: path("M20,20 C20,100 200,0 200,100");
  animation: move 3000ms infinite alternate ease-in-out;
  width: 40px;
  height: 40px;
  background: cyan;
}

@keyframes move {
  0% {
    offset-distance: 0%;
  }
  100% {
    offset-distance: 100%;
  }
}
```
