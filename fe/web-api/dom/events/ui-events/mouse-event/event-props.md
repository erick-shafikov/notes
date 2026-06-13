свойства event:

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

# clintX, clientY

clintX, clientY – координаты курсора в момент клика относительно DOM Относительно окна (без учета прокрутки)

<!--  -->

- ctrlKey - нажат ли ctrl
- metaKey - нажата ли meta
- movementX - координата x относительно последней позиции
- movementY - координата y относительно последней позиции
- offsetX - относительно границы узла
- offsetY

# pageX, pageY

MouseEvent.clientX MouseEvent.clientY - указывают на положение курсора относительно всего документа

при position:

- fixed – отсчет от верхнего левого угла окна (window) clientX, clientY
- absolute – отсчет от верхнего левого ула документа pageX, pageY. При прокрутке clientY меняется pageY

<!--  -->

- relatedTarget - второстепенная цель

# screenX

относительно экрана

<!--  -->

- shiftKey - зажат ли shift
- which - Получение информации о кнопки
- - event.which == 1 ЛКМ,
- - event.which == 2 СКМ,
- - event.which == 3 ПКМ
- mozInputSource
- webkitForce - давление при клике
- x, y - clintX, clientY

```html
<!-- окно с формой ввода с начальным текстом наведи на меня…, в котором при
наведении отображаются координаты Отключаем выделение -->
<input
  onmousemove="this.value.clientX + ':' + event.clientY"
  value="Наведи на меня мышь"
/>
```

методы:

- MouseEvent.getModifierState() - вернет состояние
- MouseEvent.initMouseEvent()

константы:

- MouseEvent.WEBKIT_FORCE_AT_MOUSE_DOWN
- MouseEvent.WEBKIT_FORCE_AT_FORCE_MOUSE_DOWN
