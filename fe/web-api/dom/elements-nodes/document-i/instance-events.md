# события

- afterscriptexecute - Non-standard Deprecated - окончание работы скрипта
- beforeonload/unload – покидает страницу
- beforescriptexecute - Non-standard Deprecated - старт работы скрипта
- copy - ClipboardEvent - при копировании
- cut
- DOMContentLoaded - документ загружен без ожидания стилей, изображений, фреймов. браузер полностью загрузил HTML, но без внешних ресурсов
- fullscreenchange - переход в fullscreen режим
- fullscreenerror - если браузер не умеет в fullscreen
- load – Браузер загрузил внешние ресурсы
- paste - при вставки
- pointerlockchange - заблокирован ли указатель
- pointerlockerror
- prerenderingchangeExperimental - запускается для предварительно отрисованного документа
- readystatechange - при изменении readyState статуса документа
- scroll - при прокрутке страницы

```js
// Источник: http://www.html5rocks.com/en/tutorials/speed/animations/

let last_known_scroll_position = 0;
let ticking = false;

function doSomething(scroll_pos) {
  // Делаем что-нибудь с позицией скролла
}

window.addEventListener("scroll", function (e) {
  last_known_scroll_position = window.scrollY;

  if (!ticking) {
    window.requestAnimationFrame(function () {
      doSomething(last_known_scroll_position);
      ticking = false;
    });

    ticking = true;
  }
});
```

- scrollend - документ пролистан
- scrollsnapchange - Experimental - прокручен контейнер
- scrollsnapchanging- Experimental
- securitypolicyviolation - при нарушении CSP
- selectionchange - при изменении выделения
- transitioned – CSS анимация завершилась
- visibilitychange - при смене видимости вкладки
