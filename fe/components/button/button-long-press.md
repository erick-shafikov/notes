# кнопка с нажатием в 200мс

реализация через js

```html
<button id="btn">Click / Long Press</button>
```

```js
//получаем элементы
const btn = document.getElementById("btn");
const clickHandler = () => console.log("Сlick Handler");
const longPressHandler = () => console.log("Long Press Handler");

let timerId, longPressed;

//таймер по нажатию
btn.onmousedown = () => {
  longPressed = false;
  timerId = setTimeout(() => {
    longPressed = true;
    // прошло 200мс - запускаем обработчик
    longPressHandler();
  }, 200);
};

//старт обработчика на клик
btn.onclick = () => {
  if (!longPressed) clickHandler();
  // сброс предыдущего
  clearTimeout(timerId);
};

//старт обработчика на потерю фокуса
btn.onmouseleave = () => {
  // сброс предыдущего
  clearTimeout(timerId);
};
```

с помощью css

```scss
#btn {
  //анимация цвета как обратная связь
  transition: background-color 1s 200ms; /* animation for long press */
}

// как нажали на кнопку
#btn:active {
  animation-name: interruptClick;
  //задержка анимации на 200
  animation-delay: 200ms;
  animation-fill-mode: forwards;
  //стили для анимации
  background-color: PaleTurquoise; /* styles for long press */
}

//потеря фокуса - сброс анимации
#btn:not(:hover) {
  animation-play-state: paused;
}

@keyframes interruptClick {
  to {
    pointer-events: none;
  }
}
```

```js
btn.onclick = () => console.log("Click Handler");

// обработчик сработает по окончанию анимации
btn.onanimationend = () => console.log("Long Press Handler");
```
