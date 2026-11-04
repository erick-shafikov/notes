Worklet - это упрощенная версия worker-ов для взаимодействия с низкоуровневыми апи как отображения, так и звука

# AudioWorklet

для обработки аудио

# AnimationWorklet

для скролл и других анимаций

# LayoutWorklet

для определения кастомных элементов

# SharedStorageWorklet

для кросс-доменных операций обмена информации

```js
// paint.js
class CirclePainter {
  paint(ctx, size) {
    ctx.fillStyle = "blue";

    ctx.beginPath();

    ctx.arc(size.width / 2, size.height / 2, 30, 0, Math.PI * 2);

    ctx.fill();
  }
}

registerPaint("circle", CirclePainter);
// main.js
CSS.paintWorklet.addModule("/paint.js");
```

```scss
// style.css
.element {
  background-image: paint(circle);
}
```
