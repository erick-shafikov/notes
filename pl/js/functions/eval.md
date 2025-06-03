# Eval

- возвращаемое значение string | undefined

```js
// Синтаксис
let result = eval("code");
// Коду через eval доступны внешние переменные
let a = 1;
function f() {
  let a = 2;
  eval("alert(a)");
}
f();
let x = 5;
eval("x=10");
alert(x); //10
// Внутри eval переменные не видны, так как у eval свое лексическое окружение Если код внутри eval не использует внешние переменные, то лучше вызывать как

window.eval;

let x = 1;

{
  let x = 5;
  window.eval("alert(x)"); //1
}
// если нужны локальные переменные, то лучше использовать new Functions
```
