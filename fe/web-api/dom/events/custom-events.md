# Генерация пользовательских событий

# Конструктор Event

let event = new Event(type [, options]);

type – тип события например click или придуманный
options:

- bubbles: true/false – если true тогда событие всплывает
- cancelable – можно отменить действия по умолчанию
- composed – событие всплывает за пределы shadow DOM

по умолчанию все false {bubbles: false, cancelable: false}

# dispatchEvent

После того как объект создан мы должны вызывать его на элементе

elem.dispatchEvent(event)

```html
<button id="elem" onclick="alert('click')">Авто клик</button>

<script>
  let event = new Event("click");
  elem.dispatchEvent(event);
</script>

<!-- event.isTrusted – для проверки событий Событие ниже позволяет сгенерировать -->
<!-- событие, которое выводит привет от тэга, фишка в том, что -->
<h1 id="elem">Привет из кода!</h1>

<script>
  document.addEventLIstener("hello", function (event) {
    //событие hello нестандартное
    //добавляем наше событие !!только через addEventListener document.onhello не сработает
    alert("Привет от" + event.target.tagName); //Привет от h1
  });

  let event = new Event("hello", { bubbles: true }); //активируем всплытие
  elem.dispatchEvent(event);
</script>
```

# MouseEvent, KeyboardEvent

Для некоторых событий есть специальные конструкторы

```js
let event = new MouseEvent("click", {
  //стандартное свойство в котором можно задать свойство clientX
  bubbles: true,
  cancelable: true,
  clientX: 100,
  clientY: 100,
});

alert(event.clientX); //100

let event = new Event("click", {
  //в new Event нет свойства clientX
  bubbles: true,
  cancelable: true,
  clientX: 100,
  clientY: 100,
});

alert(event.clientX); //undefined
```

# Пользовательские свойства

при генерации пользовательских событий следует использовать new CustomEvent, так как у него есть аргумент details

```html
<h1 id="elem">Привет</h1>

<script>
  elem.addEventListener("hello", function (event) {
    //нестандартное событие hello выводит detail.name
    alert(event.detail.name);
  });

  elem.dispatchEvent(
    new CustomEvent("hello", {
      details: { name: "Вася" }, //устанавливаем в специальный объект details свойство name
    })
  );
</script>
```

# event.preventDefault()

При вызове event.preventDefault() elem.dispatchEvent(event) возвращает false
Пример скрывающийся элемент

```html
<pre id="elem">Элемент</pre>
//тег для сохранности пробелов в примере был кролик из символов
<button onclick="hide()">Hide</button>

<script>
  function hide() {
    //функция, которая создает событие hide
    let event = new CustomEvent("hide", {
      //создаем свое событие, так как все флагу false, для возможности event.preventDefault() нужно установить свойство
      cancelable: true,
    });

    if (!elem.dispatchEvent(event)) {
      //при вызове preventDefault() вернет false
      alert("действие отменено обработчиком!");
    } else {
      elem.hidden = "true";
    }
  }

  elem.addEventLIstener("hide", function (event) {
    //добавим hide
    if (confirm("вызвать preventDefault?")) {
      //при положительном ответе скрытие объект остановится и выйдет оповещение о том, что действие отменено обработчиком, при отрицательном элемент скроется
      event.preventDefault();
    }
  });
</script>
```

# Вложенные события обрабатываются синхронно

синхронная обработка вложенных событий выполнится "1", "вложенное событие" , "2"

```html
<button id="menu">Меню</button>
<script>
  menu.onclick = function() {
    alert(1); //1
  }
  menu.dispatchEvent(new CustomEvent("menu-open", {
    bubbles: true
  }));
    alert(2); //3

document.addEventLIstener("menu-open", () +> alert("вложенное событие")) //2
<script>

```

Выполнится «1», «2», «вложенное событие»

```html
<button id="menu">Меню</button>
<script>
  menu.onclick = function() {
    alert(1); //1
  }

  setTimeout(() => menu.dispatchEvent(new CustomEvent("menu-open". {
    bubbles: true
  })));

alert(2); //3

document.addEventLIstener("menu-open", () => alert("вложенное событие")) //2
<script>

```

# использование Event вместо CustomEvent

что бы не использовать details и не использовать в ts

```ts
type CustomEvent<MyDetailType>;
```

```ts
export class MyEvent extends Event {
  static readonly eventName = "my-event";

  readonly foo: number;
  readonly bar: string;

  constructor(foo: number, bar: string) {
    super(MyEvent.eventName, { bubbles: true, composed: true });
    this.foo = foo;
    this.bar = bar;
  }
}

someElement.addEventListener(MyEvent.eventName, (e: MyEvent) => {
  // Выглядит намного чище.
  const { foo, bar } = e;
  // ...
});

declare global {
  interface GlobalEventHandlersEventMap {
    "my-event": MyEvent;
  }
}
```
