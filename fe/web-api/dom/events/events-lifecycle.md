# Всплытие и погружение

3 фазы прохода события:

- фаза погружения (capturing phase) – событие идет сверху вниз
- фаза цели (target phase) – событие достигло целевого элемента
- фаза всплытия (bubbling stage) – событие начинает всплывать

Всплытие – срабатывание обработчиков события на элементе, потом на его родителе и так далее, идет до window. принцип: когда событие происходит на элементе, обработчики срабатывают сначала на нем, потом на его родителе и так далее

```html
<form onclick="">
     <!-- 3 потом тут form -->
   
  <div onclick="">
       <!-- 2 потом тут div -->
       
    <p onclick="">
         <!-- 1 При клике здесь, сработает сначала здесь </p> -->
       
    </p>
  </div>
</form>
```

# event.target

самый глубокий элемент, который вызывает событие доступен через event.target. event.target – это целевой элемент, на котором произошло событие в процессе всплытия он неизменен. this – это текущий элемент, до которого дошло всплытие, на нем сейчас обработчик

внутри обработчика form.onclick this = event.currentTarget всегда будет элемент form, так как обработчик сработал на ней. event.target будет содержать ссылку на конкретный элемент внутри формы, на котором произошел клик. совпадают, когда клик произошел непосредственно на элементе

# Прекращение всплытия

Что бы остановить всплытие нужно вызвать метод event.stopPropagation(), событие будет вызвано только на самом элементе.

```html
<body onclick='alert("всплытие до сюда не дойдет")'>
  <button onclick="event.stopPropagation()">Кликни меня</button>
</body>
```

event.stopImmediatePropagation()
если у элемента есть несколько обработчиков одного события, то даже при прекращении всплытия все они будут выполнены

# Погружение

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

Поведение

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

# bps. Применение делегирования

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
