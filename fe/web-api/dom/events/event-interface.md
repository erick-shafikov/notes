# интерфейс события Event

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

свойства:

- bubbles ⇒ boolean всплыло ли событие или нет
- cancelBubble ⇒ boolean если установить true не будет всплывать
- cancelable ⇒ boolean можно ли отменить
- composed ⇒ boolean может ли всплывать между shadow dom и обычным
- currentTarget - ссылка не элемент на котором обрабатывается событие
- deepPath - массив dom узлов, на которых сработало событие
- defaultPrevented - было ли вызвано event.preventDefault()
- eventPhase - фаза события
- explicitOriginalTarget - первоначальный целевой элемент
- originalTarget - Первоначальный целевой объект события до перенаправлений
- scoped - всплывает ли данное событие через shadow root
- target - элемент на котором произошло событие
- timeStamp - элемент когда произошло событие
- type – тип события в примере "click"
- isTrusted - событие запущено с помощью клика мыши или скрипта

методы:

- createEvent() - создание события для дальнейшего использования createEvent()
- initEvent() - запуски события
- preventDefault() - отмена события
- stopImmediatePropagation() - отмена на фазе перехвата
- stopPropagation() - остановка события далее по Dom
