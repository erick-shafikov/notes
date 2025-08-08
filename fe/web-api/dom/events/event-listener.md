# Обработчик событий

функция, которая срабатывает, как только событие произошло

- Использование атрибута HTML, inline

```html
<!-- Использование атрибута HTML, inline -->
<input value="Нажми меня" onclick="alert('Клик!')" type="button" />

<!-- Использование функции в атрибуте HTML -->
<script>
  function countRabbits() {
    for (let i = 1; i <= 3; i++) {
      alert("кролик номер" + i);
    }
  }
</script>

<input type="button" onclick="countRabbits()" value="Считать кроликов" />

<!-- Присвоение элементу -->
<input id="elem" type="button" value="Нажми меня!" />    
<script>
  elem.onclick = function () {
    alert("Спасибо");
  };
</script>
```

- !!!Обработчик всегда хранится в свойстве DOM объекта, а атрибут – один из способов его инициализировать
- !!!Назначить более одного обработчика невозможно
- !!!при присвоении уже существующий функции DOM – свойству, нельзя ставить скобки button.onclick = func, а не button.onclick = function() т.к. присвоит результат
- !!!При присвоении в HTML, нужно ставить скобки <… onclick="func()">
- !!!В атрибутах использовать функцию а не строки
- !!!Не использовать setAttribute, так как при создании все станет строкой
- !!!Регистр DOM-свойства имеет значения

# addEventListener

```js
element.addEventListener(event, handler, {
  //дополнительный объект со свойствами
  once: false, //при true Обработчик сразу будет удален,
  capture: capturePhase, //фраза на которой должен сработать обработчик,
  passive: true, //при true указывает на то, что обработчик никогда не вызовет preventDefault()
});
// event – событие, handler – ссылка на функцию обработчик

// Удаление требует ту же функцию, не сработает:
elem.addEventListener("click", () => alert("message"));
elem.removeEventListener("click", () => alert("message"));
```

- !!! Позволяет добавить несколько обработчиков
- !!! обработчики таких свойств как DOMContentLoaded можно добавить только через addEventListener

# Объект-обработчик handleEvent

Мы можем назначить обработчиком не только функцию, но и объект при помощи addEventListener, с помощью вызова метода handleEvent

```html
<button id="elem">нажим меня</button>

<script>
  elem.addEventLIstener("click", {
    //при вызове вызывается object.handleEvent(event)
    handleEvent(event) {
      alert(event.type + "on" + event.currentTarget);
    },
  });
</script>
```

или использовать класс

```html
<button id="elem">Push me</button>
<script>
  class Menu {
    handleEvent(event) {
      switch (event.type) {
        case "mousedown":
          elem.innerHTML = "button is pushed";
          break;
        case "mouseup":
          elem.innerHTML += "and released";
          break;
      }
    }
  }

  elem.addEventListener("mouseup", menu);
  elem.addEventListener("mouseup", menu);
</script>
```

handleEvent – не обязательно должен выполнять всю работу сам, он может вызывать другие методы

```html
<button id="elem">Нажми меня</button>

<script>
  class Menu(event){
    let method = "on" + event.type[0].toUpperCase() + event.type.slice(1);
    this[method](event);


  onMousedown() {
     elem.innerHTML = "button is pushed";
  }

  onMouseup() {
    elem.innerHTML += "…end released";
  }
}

```
