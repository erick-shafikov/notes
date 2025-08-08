# Событие мыши MouseEvent

свойства:

- altKey - нажат ли alt

```html
<button id="button">Нажми Alt+Shift+Click</button>

<script>
  button.onclick = function (event) {
    if (event.altKey && event.shiftKey) {
      alert("Ура");
    }
  };
</script>
```

- button - код клавиши
- buttons - код клавиш которые были нажаты
- clintX, clientY – координаты курсора в момент клика относительно DOM Относительно окна (без учета прокрутки)
- ctrlKey - нажат ли ctrl
- metaKey - нажата ли meta
- movementX - координата x относительно последней позиции
- movementY - координата y относительно последней позиции
- offsetX - относительно границы узла
- offsetY
- pageX - относительно всего документа
- pageY -
- relatedTarget - второстепенная цель
- screenX - относительно экрана
- shiftKey - зажат ли shift
- which - Получение информации о кнопки
- - event.which == 1 ЛКМ,
- - event.which == 2 СКМ,
- - event.which == 3 ПКМ
- mozInputSource
- webkitForce - давление при клике
- x, y - clintX, clientY

```html
<input
  onmousemove="this.value.clientX + ':' + event.clientY" value="Наведи на меня мышь"
/>
<!-- окно с формой ввода с начальным текстом наведи на меня…, в котором при
наведении отображаются координаты Отключаем выделение -->

<span ondbclick="alert('dbclick')">Сделай двойной клик</span>
<!-- alert выведет текст, но также выделится слово на котором произведен двойной клик -->

<span ondbclick="alert('dbclick')" onmousedown="return false"
  >Сделай двойной клик</span
>
<!-- теперь текст не выделяется и его нельзя выделить Предотвращение копирования -->

<div oncopy='alert("Копирование запрещено"); return false;' >
  Копирования текста запрещено
  </div>
</div>
```

методы:

- MouseEvent.getModifierState() - вернет состояние
- MouseEvent.initMouseEvent()

константы:

- MouseEvent.WEBKIT_FORCE_AT_MOUSE_DOWN
- MouseEvent.WEBKIT_FORCE_AT_FORCE_MOUSE_DOWN

## Типы событий мыши

Простые события:

- mousedown/mouseup – кнопка мыши нажата/опущена
- mouseover/mouseup – курсор мыши появляется над элементом и уходит с него
- mousemove – каждое движение над этим элементом генерирует это событие
- contextmenu – ПКМ или вызов контекстного меню с клавиатуры

Комплексные события:

- click – вызывает при mousedown а затем mouseup над одним и тем же элементом
- dbclick – вызывает при двойном клике на элементе

Движение мыши События mouseover/mouseout, relatedTarget:

Для события mouseover:

- event.target – это элемент на который курсор перешел
- event.relatedTarget – это элемент с которого курсор ушел
- Для события mouseout наоборот

```html
<body>
  <!--лица-->
  <div id="container">
    <div class="smiley-green">
      <div class="left-eye"></div>
      <div class="right-eye"></div>
      <div class="smile"></div>
    </div>

    <div class="smiley-yellow">
      <div class="left-eye"></div>
      <div class="right-eye"></div>
      <div class="smile"></div>
    </div>

    <div class="smiley-red">
      <div class="left-eye"></div>
      <div class="right-eye"></div>
      <div class="smile"></div>
    </div>
  </div>
  <textarea id="log">
События будут показываться здесь!
</textarea
  >
  <script src="script.js"></script>
</body>
```

```js
container.onmouseover = container.onmouseout = handler;

function str(el) {
  //часть для вывода в log доп информации
  if (!el) return "null";

  return el.className || el.tagName;
}

function handler(event) {
  log.value +=
    event.type +
    ": " +
    "target=" +
    str(event.target) +
    ", relatedTarget= " +
    str(event.relatedTarget) +
    "\n";

  log.scrollTop = log.scrollHeight;

  if (event.type == "mouseover") {
    event.target.style = "pink";
  }

  if ((event.type = "mouseout")) {
    event.target.style.background = "";
  }
}
```

### Событие mouse при переходе не потомка

Событие mouseout генерируется в том числе, когда указатель переходит с элемента на его потомка. Визуально курсор еще на элементе но мы получим mouseout. Курсор может быть над одним элементом над самым глубоко вложенным и верхнем z-index

Событие mouseover происходящее на потомке всплывает, если на родительском элементе есть такой обработчик то он его вызовет

```js
parent.onmouseout = function (event) {
  /_event.target: внешний элемент_/;
};

parent.onmouseover = function (event) {
  /_event.target: внутренний элемент всплыло_/;
};
```

### mouseenter mouseleave

Отличия от mouseover и mouseout в том, что переходы внутри элемента на его потомки и с них не считаются
mouseenter и mouseleave не всплывают

### Делегирование

```js
// Обработчик под указателем мыши
// пример подкрашивает ячейки таблицы, причем все
table.onmouseover = function (event) {
  //закрасим
  let target = event.target;
  target.style = background = "pink";
};
table.onmouseout = function (event) {
  let target = event.target;
  target.style.background = "";
};

// пример, где будут подсвечиваться только td
let currentElem = null;
//переменная для текущего элемента
table.onmouseover = function (event) {
  //При наведении
  if (currentElem) return;
  //если элемент не выбран выйти из функции
  let target = event.target.closest("td");
  //найти среди предков td, если вложенный элемент
  if (!target) return;
  //если нет среди предков - выйти
  if (!table.contains(target)) return;
  //элемент должен быть внутри нашей таблицы
  currentElem = target; //нашли
  target.style.background = "pink"; //покрасили
};

table.onmouseout = function (event) {
  //покидаем элемент
  if (!currentElem) return; //если не определен элемент - выход
  let relatedTarget = event.relatedTarget;
  //определяем покинутый объект
  while (relatedTarget) {
    if (relatedTarget == currentElem) return;
    relatedTarget = related.parentNode;
  } //поднимаемся выше по дереву, что бы отфильтровать переход между td и span

  currentElem.style.background = ""; //сброс цвета
  currentElem = null; //обнуляем текущей элемент
};
```

### BP. улучшенная подсказка

```js
let tooltip;

document.onmouseover = function (event) {
  let anchorElem = event.target.closest("[data-tooltip]");
  //ищем предков по классу, у которого есть атрибут data-tooltip
  if (!anchorElem) return;
  tooltip = showTooltip(anchorElem, anchorElem.dataset.tooltip);
};
document.onmouseout = function () {
  if (tooltip) {
    tooltip.remove();
    tooltip = false;
  }
};

function showTooltip(anchorElem, html) {
  let tooltipElem = document.createElement("div");
  tooltipElem.className = "tooltip";
  tooltipElem.innerHTML = html;
  document.body.append(tooltipElem);

  let coords = anchorElem.getBoundingClientRect();

  let left = coords.left + (anchor.offsetWidth - tooltipElem.offsetWidth) / 2;
  if (left < 0) left = 0;
  let top = coords.top - tooltipElem.offsetHeight - 5;
  if (top < 0) {
    coords.top + anchorElem.offsetHeight + 5;
  }
  tooltipElem.style.left = left + "px";
  tooltipElem.style.top = top + "px";
  return tooltipElem;
}
```

## Drag'n'Drop с событиями мыши

```js
// Алгоритм Drag"n"Drop

ball.onmousedown = function (event) {
  let shiftX = event.clientX - ball.getBoundingClientRect().left;
  let shiftY = event.clientY - ball.getBoundingClientRect().top;

  ball.style.position = "absolute";
  ball.style.zIndex = 1000;
  document.body.append(ball);

  moveAt(event.pageX, event.pageY); //координаты курсора мыши

  function moveAt(pageX, pageY) {
    ball.style.left = pageX - shiftX + "px";
    ball.style.top = pageY - shiftY + "px";
  }

  function omMouseMove(event) {
    moveAt(event.pageX, event.pageY);
  }

  document.addEventLIstener("mousemove", onMOuseMove);
  ball.onmouseup = null;
};

ball.ondragstart = function () {
  return false;
};
```

### Цели переноса

При перетаскивании элементов, перетаскиваемый элемент находится выше остальных

```html
<div style="background: blue" onmouseover="alert('Не сработает')"></div>
<div style="background: red" onmouseover="alert('над красным')"></div>
```

С помощью document.elementFromPoint(clientX, clientY);

```js
ball.hidden = true;
let elemBelow = document.elementFromPoint(event.clientX, event.clientY); //elemBelow - элемент под мячом
ball.hidden = false;

let currentDroppable = null; //потенциальная цель переноса
function onmouseMove(event) {
  moveAt(event.pageX, event.pageY);
  ball.hidden = true; //прячем мяч
  let elemBelow = document.elementFromPoint(event.clientX, event.clientY);
  //находим самый спрятанный элемент
  ball.hidden = false; //возвращаем мяч

  if (!elemBelow) return; // за пределами окна - выход
  let droppableBelow = elemBelow.closest(".droppable"); //находим элемент для переноса с классом droppable
  if (currentDroppable != droppableBelow) {
    if (currentDroppable) {
      leaveDroppable(currentDroppable); //функция подсветки, при выходе из объекта
    }
    currentDroppable = droppableBelow;
    if (currentDroppable) {
      enterDroppable(currentDroppable); //функция попадания на элемент
    }
  }
}
```

## Drag'n'Drop с помощью drag'n'drop api

События по умолчанию - dragenter, dragover, dragstart

Нужно добавить аттрибут draggable на контейнер с элементом

```js
draggableElement.addEventListener("dragenter", (e) => {
  e.preventDefault();
});

draggableElement.addEventListener("dragover", (e) => {
  e.preventDefault();
});

draggableElement.addEventListener("drop", (e) => {
  e.preventDefault();
  e.dataTransfer.setData('text/plain')
//коллбек на выполнение при сбросе
  functionThatFiresWhenDragAndDrop()
});


// вешаем на контейнер
document.querySelectorAll().forEach((row) => {
  row.addEventListener('dragstart' (e) => {
    e.dataTransfer.setData('text/plain')
  })
})


```

### BP. Slider

```html
<body>
   
  <div id="slider" class="slider">
        
    <div class="thumb"></div>
      
  </div>
   
  <script>
    let thumb = slider.querySelector(".thumb");
    //находим thumb
    thumb.onmousedown = function (event) {
      event.preventDefault(); //отменяем выделение
      let shiftX = event.clientX - thumb.getBoundingClientRect().left;
      //убираем подпрыгивание слайдера
      document.addEventListener("mousemove", onMouseMove); //добавляем передвижение
      document.addEventListener("mouseup", onMouseUp);

      function onMouseMove(event) {
        let newLeft =
          event.clientX - shiftX - slider.getBoundingClientRect().left;
        if (newLeft < 0) {
          newLeft = 0;
        } //левая граница
        let rightEdge = slider.offsetWidth - thumb.offsetWidth;
        if (newLeft > rightEdge) {
          newLeft = rightEdge;
        } //правая граница
        thumb.style.left = newLeft + "px";
        //окончательное перемещение
      }
      function onMouseUp() {
        document.removeEventListener("mouseup", onMouseUp);
        document.removeEventListener("mousemove", onMouseMove);
      }
    };
    thumb.ondragstart = function () {
      return false;
    };
  </script>
</body>
```
