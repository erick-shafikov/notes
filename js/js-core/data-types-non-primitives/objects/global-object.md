# Global object

```js
// Глобальный объект предоставляет переменные и функции в любом месте программы  в браузере глобальные объекты объявляются с помощью var
var gVar = 5;
alert(window.gVar); //5

// можно записывать очень важные свойства
windows.currentUser = {
  name: "John",
};
alert(currentUser.name); // John
alert(windows.currentUser.name); //John

// TG;
const a = 1;
delete a; //false
console.log(a); //1
this.b = 4;
delete b; //true
console.log(b); //undefined
```
