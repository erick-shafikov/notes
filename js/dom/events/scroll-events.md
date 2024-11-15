# Прокрутка

Событие scroll

```js
window.addEventListener("scroll". function() {
document.getElementById("showScroll").innerHTML = pageYoffset + "px";
});
```

Предотвращение прокрутки

Не получится предотвратить прокрутку в обработчике onscroll используя preventDefault(), но можно на событиях keydown и для клавиш pageUp pageDown. Самый надежный способ использовать css свойство overflow
