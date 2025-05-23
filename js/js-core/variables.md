# Объявление переменных

```js
$, _; //допустимы в именах переменных
```

При объявлении переменной через var, её видно вне блоков if и while

переменные объявленный с помощью var обрабатываются в начале выполнения кода  
у var не имеет облачной видимости, инициализируется в начале выполнения кода, если объявлена внутри функции, то становится локальной переменной

Разница между let и var

- Переменная var доступна за пределами блоков {} в отличие от let и const
- При объявлении внутри функции разницы нет
- Var допускает повторное объявление функции
- Инициализируется в начале, присваивается при присваивании
- var добавляет переменную в window

```js
function myFunc() {
  var a = 1;
  if (a == 1) {
    var b = 2; //var создает переменную в контексте текущей функции а не в контексте текущего scope
  }
  return b;
}
console.log(myFunc()); //2
```

# Хостинг

- JS сначала объявляет, потом присваивает переменные, если обратиться с переменной до ее присвоения, то результатом будет не ошибка, а undefined в нестрогом режиме, в строгом будет ошибка, в ES6 использование переменной до объявления выдает ошибку.

- Функции FD позволяет вызывать функции до инициализации, FE приведет к ошибке
- var позволяет обращаться к переменной до объявления, let и const приведут к ошибке

```js
// var ------------------------------------------------------------------

var a;
console.log("The value of a is " + a); //Значение переменной a undefined
console.log("The value of c is " + c); //Значение переменной c undefined
var c;
console.log(x === undefined); // true
var x = 3;

var myvar = "my value";

(function () {
  console.log(myvar); // undefined
  var myvar = "local value";
})();

// ----------------------------------------------------------------------

console.log("The value of b is " + b); //Uncaught ReferenceError: b не определена

console.log("The value of x is " + x); //Uncaught ReferenceError: x не определена
let x;

function do_something() {
  console.log(foo); // ReferenceError
  let foo = 2;
}

// function -------------------------------------------------------------

/* Определение функции */
foo(); // "bar"

function foo() {
  console.log("bar");
}

/* Определение функции через выражение */
baz(); // TypeError: baz is not a function

var baz = function () {
  console.log("bar2");
};
```

# Область видимости

```js
if (true) {
  var x = 5;
}
console.log(x); // 5

if (true) {
  let y = 5;
}
console.log(y); // ReferenceError
```

# со скобками

```js
var x = 5;
var y = 0;

let (x = x+10, y = 12) {
  console.log(x+y); // 27
}

console.log(x + y); // 5

```
