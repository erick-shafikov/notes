# Обработчик событий

функция, которая срабатывает, как только событие произошло

# Использование атрибута HTML, inline

```html
<input value="Нажми меня" onclick="alert('Клик!')" type="button" />
```

# Использование функции в атрибуте HTML

```html
<script>
  function countRabbits() {
    for (let i = 1; i <= 3; i++) {
      alert("кролик номер" + i);
    }
  }
</script>

<input type="button" onclick="countRabbits()" value="Считать кроликов" />
```

# Присвоение элементу

```html
<input id="elem" type="button" value="Нажми меня!" />    
<script>
  elem.onclick = function () {
    alert("Спасибо");
  };
</script>
```

!!!Обработчик всегда хранится в свойстве DOM объекта, а атрибут – один из способов его инициализировать
!!!Назначить более одного обработчика невозможно
!!!при присвоении уже существующий функции DOM – свойству, нельзя ставить скобки
button.onclick = func, а не button.onclick = function() т.к. присвоит результат
!!!При присвоении в HTML, нужно ставить скобки <… onclick="func()">
!!!В атрибутах использовать функцию а не строки
!!!Не использовать setAttribute, так как при создании все станет строкой
!!!Регистр DOM-свойства имеет значения

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

# Объект события

Когда происходит событие, браузер создает объект события записывает в него детали и передает его в качестве аргумента функцию-обработчику

```js

<input type="button" value="Push me" id="elem">

<script>
  elem.onclick = function(event){
    alert(event.type + "on" + event.currentTarget);
    alert("coords:" + event.clientX + ":" + event.clientY);
  };
<script>
```

некоторые свойства:

- event.type – тип события в примере "click"
- event.currentTarget – элемент на котором сработал обработчик
- event.clintX и event.clientY – координаты курсора в момент клика

# Объект-обработчик handleEvent

Мы можем назначить обработчиком не тольео функцию, но и объект при помощи addEventListener, с помощью вызова метода handleEvent

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
<button id=""elem"">Push me</button>
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

# Всплытие и погружение

3 фазы прохода события:

- фаза погружения (capturing phase) – событие идет сверху вниз
- фаза цели (target phase) – событие достигло целевого элемента
- фаза всплытия (bubbling stage) – событие начинает всплывать

Всплытие – срабатывание обработчиков события на элементе, потом на его родителе и так далее, идет до window. принцип: когда событие происходит на элементе, обработчики срабатывают сначала на нем, потом на его родителе и так далее

<form onclick="">
   <!-- /3/ потом тут -->
   <div onclick="">
   <!-- /2/ потом тут -->
   <p onclick="">
   <!-- /1/ При клике здесь, сработает сначала здесь</p> -->
  </div>
</form>

## event.target

самый глубокий элемент, который вызывает событие доступен через event.target, отличия от event.currentTarget
event.target – это целевой элемент, на котором произошло событие в процессе всплытия он неизменен
this – это текущий элемент, до которого дошло всплытие, на нем сейчас обработчик

внутри обработчика form.onclick
this = event.currentTarget всегда будет элемент <form>, так как обработчик сработал на ней
event.target будет содержать ссылку на конкретный элемент внутри формы, на котором произошел клик

совпадают, когда клик произошел непосредственно на элементе

## Прекращение всплытия

Что бы остановить всплытие нужно вызвать метод event.stopPropagation(), событие будет вызвано только на самом элементе.

```html
<body onclick='alert("всплытие до сюда не дойдет")'>
  <button onclick="event.stopPropagation()">Кликни меня</button>
</body>
```

event.stopImmediatePropagation()
если у элемента есть несколько обработчиков одного события, то даже при прекращении всплытия все они будут выполнены

## Погружение

поймать на стадии погружения Погружение или перехват – первая фаза при срабатывания события, но будет доступна при добавлении дополнительного параметра в addEventLIstener(…,…{capture:true});

elem.addEventListener(…, {capture: true}) или elem.addEventLIstener(…,true)

# Делегирование событий

- вешаем обработчик на контейнер
- В обработчике проверим исходный элемент на event.target
- Если событие произошло внутри нужного элемента, то обрабатываем его

Делегирование события – прием при котором можно определить обработчик на родительский элемент, чтобы не добавлять одинаковый обработчик на множество вложенных элементов

```html
<table>
  <tr>
    <th colspan="3">…</th>
  </tr>
  <tr></tr>
</table>

 
<script>
  let selectedTd;
  table.onclick = function (event) {
    //эта часть не будет работать если кликнуть на вложенный <strong>
    let target = event.target; //определяем где был клик, так как event.target – самый глубокий элемент, присвоим переменной target event.target события
    if (target.tagName != "TD") return; //если не на TD тогда не интересует
    highlight(target);
  }; //подсветим TD, отправим в качестве аргумента ячейку target

  function highlight(td) {
    if (selectedTd) {
      //убрать подсветку, если есть, так как это условие выполнится, если на предыдущем шаге selectedTd была определена
      selectedTD.classlist.remove("highlight");
    }
    selectedTd = td; //selectedTd = target = event.target
    selectedTd.classList.add("highlight"); //подсветить новый td
  } // исправим

  table.onclick = function (event) {
    let td = event.target.closest("td"); //возвращает ближайшего предка, соответствующего селектору
    if (!td) return; // если event.target не содержится внутри элемента <td> то вызов вернет null
    if (!table.contains(td)) return; //проверяем относится ли td к нашей таблице
    highlight(td);
  };
</script>
```

## Применение делегирования

Меню с разными кнопками save, load, search и объект с этими методами, как их состыковать

```html
<div id="menu">
  <button data-action="save">нажмите. чтобы сохранить</button>  
  <button data-action="load">Загрузить</button>
  <button data-action="search">Поиск</button>
</div>

<script>
  class Menu {
    constructor(elem) {
      this._elem = elem;
      elem.onclick = this.onClick.bind(this); //привязка к контексту так как иначе this будет ссылать на DOM элемент, а не на объект menu. так же определяет один из методов
    }
    save() {}
    load() {}
    search() {}
    onClick(event) {
      //определитель метода
      let action = event.target.dataset.action; //dataset.action = "save" или "load" или "search"
      if (action) {
        this[action]();
      }
    }
  }
  new Menu(menu);
</script>
```

## Поведение

- Элементу присваивается атрибут описывающий его поведение
- Ставится обработчик на документ, который ловит все клики Счетчик Счетчик

```html
<input type="button" value="1" data-counter />
<!-- Еще счетчик: -->
<input type="button" value="2" data-counter />

<script>
  document.addEventListener("click", function(event)) {
    if(event.target.dataset.counter != undefined) { //если есть атрибут
      event.target.value++;
    }
  };
</script>
```

```html
<button data-toggle-id="subscribe-mail">Показать форму подписки</button>
<form id="subscribe-mail" hidden>Ваша почта: <input type="email" /></form>

<script>
  document.addEventLIstener("click", function (event) {
    let id = event.target.dataset.toggledId; //присваиваем id значение subscribe-mail
    if (!id) {
      return;
    }
    let elem = document.getElementById(id); //Найти элемент с id === subscribe-mail
    elem.hidden = !elem.hidden;
  });
</script>
```

# Действия браузера по умолчанию

Многие действия влекут за собой действия браузера:

- клик по ссылке инициирует переход на другой URL
- нажатие кнопки отправить – отправляет на сервер
- зажатие кнопки над текстом – выделение текста

## Отмена действий браузера

- event.preventDefault() – для отмены действия браузера
- Вернуть false из обработчика

```html
<a href="/" onclick="return false">Нажми здесь</a>
<!-- обе ссылки которые прерывают действие -->
<a href="/" onclick="event.preventDefault()">здесь</a>
```

!!!Возвращать true не нужно, значение, которое возвращает обработчик – игнорируется

## Меню

Все кнопки являются ссылками, но при использовании ПКМ и открыть в новом окне это работать не будет

```html
<ul id="menu" class="menu">
  <li><a href="/html">HTML</a></li>
  <li><a href="/javascript">JavaScript</a></li>
  <li><a href="/css">CSS</a></li>
</ul>

<script>
  //необходимо обработать клики в JS, а стандартное действие переход – отменить
  menu.onclick = function (event) {
    if (event.target.nodeName != "a") return;

    let href = event.target.getAttribute("href"); //получить ссылку
    alert(href); //что-то сделать с ней

    return false; //отменяем стандартное действие – переход по ссылки
  };
</script>

<!-- фокус работает -->
<input value="Фокус работает" onfocus="this.value=''" />
<!-- фокус не работает -->
<input onmousedown="return false" onfocus="this.value" value="Кликни меня" />
```

## Опция passive

passive: true сигнализирует для addEventListener что обработчик не собирается выполнять preventDefault()

## event.defaultPrevented

event.defaultPrevented

свойство event.defaultPrevented установлено в true, если действие по умолчанию было предотвращено
false = если нет

По умолчанию браузер при событии contextmenu – ПКМ показывает контекстное меню, можно отменить

```html
<button>Правый клик вызывает контекстное меню браузера</button>
<button contextmenu="alert('Наше меню'); return false">
  Правый клик вызывает наше контекстное меню
</button>
```

Контекстное меню для всего документа

```html
<p>Правый клик здесь вызывает контекстное меню документа</p>
<button id="elem">Правый клик здесь вызывает контекстное меню кнопки</button>

<script>
  elem.oncontextmenu = function (event) {
    event.preventDefault();
    alert("Menu");
  };

  document.oncontextmenu = function (event) {
    event.preventDefault();
    alert("menu");
  };
</script>
```

в данном примере, из-за всплытия, при клики на вложенный элемент вызывается обработчик и для document

Добавим event.stopPropagation();

```html
<p>Правый клик вызывает меню документа</p>
<p>
  <button id="elem">
    Правый клик вызывает меню кнопки (добавлен event.stopPropagation)
  </button>

  <script>
    elem.oncontextmenu = function (event) {
      event.preventDefault();
      event.stopPropagation();
      alert("меню кнопки");
    };

    document.oncontextmenu = function (event) {
      event.preventDefault();
      alert("document menu");
    };
  </script>

  Минус – мы навсегда запретили доступ к информации о правых кликах для любого
  внешнего кода
</p>

<p>Правый клик вызывает меню документа</p>
<p>
  <button id="elem">Правый клик вызывает меню кнопки</button>
  <script>
    elem.oncontextmenu = function (event) {
      event.preventDefault();
      alert("контекстное меню кнопки");
    };
    document.oncontextmenu = function (event) {
      if (event.defaultPrevented) return;
      event.preventDefault();
      alert("document menu");
    };
  </script>
</p>
```

# Генерация пользовательских событий

## Конструктор Event

let event = new Event(type [, options]);

type – тип события например click или придуманный
options:

- bubbles: true/false – если true тогда событие всплывает
- cancelable – можно отменить действия по умолчанию
- composed – событие всплывает за пределы shadow DOM

по умолчанию все false {bubbles: false, cancelable: false}

## dispatchEvent

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

## MouseEvent, KeyboardEvent

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

## Пользовательские свойства

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
      detail: { name: "Вася" }, //устанавливаем в специальный объект detail свойство name
    })
  );
</script>
```

## event.preventDefault()

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

## Вложенные события обрабатываются синхронно

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

# DOMContent loaded/load/beforeonload

- DOMContentLoaded – браузер полностью загрузил HTML, но без внешних ресурсов
- load – Браузер загрузил внешние ресурсы
- beforeonload/unload – покидает страницу

События анимаций:

- transitioned – CSS анимация завершилась
