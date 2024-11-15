document.timeline - позволяет замерять время от открытия окна до закрытия. Оно позволяет проводить манипуляции с анимацией

Определяют

- Animation Object - позволяет запускать, останавливать, искать анимацию
- Animation Effect - основные моменты анимации

# Создание анимации

Анимация вида

```scss
@keyframes moveAndRotate {
  0% {
    transform: translateX(0);
  }
  80% {
    background-color: blue;
  }
  100% {
    transform: translateX(calc(100vw - 100px)) rotate(360deg);
    background-color: crimson;
  }
}
```

```html
<body>
  <div class="square"></div>
</body>
```

Может быть преобразована

```js
document.addEventListener("DOMContentLoaded", () => {
  const element = document.querySelector(".animated-element");

  //создаем эффект
  const animationKeyframes = new KeyframeEffect(
    // элемент
    element,
    // ключевые кадры задаем в виде массива
    [
      { transform: " translateX(0)" },
      {
        backgroundColor: "blue",
        offset: 0.8,
      },
      {
        transform: "translateX(calc(100vw - 100px)) rotate(360deg)",
        backgroundColor: "crimson",
      },
    ],
    // настройки
    {
      duration: 3000,
      delay: 1000,
      direction: "alternate",
      fill: "both",
      iterations: Infinity,
      easing: "linear",
      composite: "add",
    }
  );

  // создаем объект анимации
  const animation = new Animation(
    animationKeyframes,
    // отсчет времени
    document.timeline
  );

  //запуск
  animation.play();
});
```

Сокращенная запись - вызов метода animate на самом элементе, запускается автоматически

```js
document.addEventListener("DOMContentLoaded", () => {
  const animatedElement = document.querySelector(".animated-element");

  //
  const squareAnimation = animatedElement.animate(
    [
      { transform: " translateX(0)" },
      {
        backgroundColor: "blue",
        offset: 0.8,
      },
      {
        transform: "translateX(calc(100vw - 100px)) rotate(360deg)",
        backgroundColor: "crimson",
      },
    ],
    {
      duration: 3000,
      delay: 1000,
      direction: "alternate",
      fill: "both",
      iterations: Infinity,
      easing: "linear",
      composite: "add",
      // timeline: document.timeline,
    }
  );
});
```

Можно отталкиваться от свойств. которые нужно анимировать

```js
document.addEventListener("DOMContentLoaded", () => {
  const animatedElement = document.querySelector(".animated-element");

  const animation = animatedElement.animate(
    {
      transform: [
        "translateX(0)",
        "translateX(calc(100vw - 100px)) rotate(360deg)",
      ], //offset [0,1]
      backgroundColor: ["gold", "blue", "crimson"], //offset [0, 0.5, 1]
      offset: [0, 0, 3, 1], //изменить offset
      easing: ["ease-in"],
      composite: ["add", "replace", "add"],
    },
    {
      duration: 3000,
      delay: 1000,
      direction: "alternate",
      fill: "both",
      iterations: Infinity,
      easing: "linear",
      composite: "add",
      // timeline: document.timeline 1.1[2],
    }
  );

  squareAnimation.play();
});
```

# Методы анимации

```js
animation.play();
animation.pause();
animation.cancel();
animation.reverse();
animation.finish();
animation.updatePlaybackRate(); //скорость воспроизведения
animation.currentTime = 1000; //мс сеттер установить стартовую позицию для анимации
animation.startTime = 1000; //мс сеттер установить стартовую время на временном промежутке timeline: document.timeline 1.2[2], при отрицательных значениях начнет анимацию с момента + animation.startTime
await animation.readyState; //возвращает промис
await animation.finished; //возвращает промис
```

# Объект effect

```js
animation.effect.setKeyFrame([]); //позволяет изменить кадры анимации
animation.effect.updateTiming({
  //изменяет временные параметры
  iterations: Infinity,
  // только числа
  duration: 2,
});
```

```js
document.getAnimations(); //вернет все анимации
element.getAnimations({ subtree: true }); //вернет все анимации (так же и вложенные)
```

# Commit style

При значении fill: "both" окончательные стили впишет в Inline стиль элемента

```js
animation.addEventLIstener("finish", () => {
  // вписываем стили
  animation.commitStyle();
  // очищаем

  animation.cancel();
});
```

```js
// позволяет очистить анимацию
animation.persist();
```
