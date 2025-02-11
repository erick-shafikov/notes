# Arrow. Функции стрелки

```js
// создаёт функцию func с аргументами arg1..argN и вычисляет  expression с правой стороны с их использованием, возвращая  результат:
let func = (arg1, arg2, ...argN) => expression;

// Пример:
let age = prompt("Сколько Вам лет?", 18);
let welcome = age < 18 ? () => alert("Привет") : () => alert("Здравствуйте!");
welcome(); // Если у нас только один аргумент, то круглые скобки вокруг  параметров можно опустить, сделав запись ещё короче:
// тоже что и
// let double = function(n) { return n * 2 }  let double = n => n * 2;
alert(double(3)); // 6 // Если нет аргументов, указываются пустые круглые скобки:
let sayHi = () => alert("Hello!"); // Если несколько инструкций, то заключить в скобки {}
let sayBye = () => {
  alert("Hello!");
  alert("Bye!");
};
```

## this

у стрелочных функций нет this, если идет обращение  к this, То его значение берется снаружи

```js
let group = {
  title: "Our Group",
  students: ["john", "pete", "alice"],
  showList() {
    this.students.forEach((student) => alert(this.title + ":" + student));
  },
};
```

### Частичное применение без контекста

у стрелочных функций нет this

```js
let group = {
  title: "Our Group",
  students: ["John", "Pete", "Alice"],
  showList() {
    // работает this.title = group.title
    this.students.forEach((students) => alert(this.title + ":" + student));
  },

  showList() {
    // не сработает title = undefined,
    // так как forEach выполняет с this = undefined
    this.students.forEach(function (student) {
      alert(this.title + ":" + student);
    });
  },
};

group.showList();
```

!!!Нельзя использовать с new
!!!При использовании с bind, как обычная переменная берется из внешнего ЛО

Стрелочные функции не имеют arguments

```js
group.showList();
function defer(f, ms) {
  return function () {
    setTimeout(() => f.apply(this, arguments), ms);
  };
}
function sayHI(who) {
  alert(`Hello, ` + who);
}
let sayHiDeferred = defer(sayHi, 2000);
sayHiDeferred("John");

// TG повторное объявление после параметра
var a = 1;
function foo(x = 2) {
  let x = 5;
  console.log(x);
}
foo(); // Uncaught SyntaxError: Identifier 'x' has already been declared
```

# сохранение контекста

```js
function Person() {
  // Конструктор Person() определяет `this` как самого себя.
  this.age = 0;

  setInterval(function growUp() {
    // Без strict mode функция growUp() определяет `this` как global object, который отличается от `this` определённого конструктором Person().
    this.age++;
  }, 1000);
}

var p = new Person();

// решение 1 запомнить контекст выполнения
function Person() {
  var self = this; // Некоторые выбирают `that` вместо `self`.
  // Выберите что-то одно и будьте последовательны.
  self.age = 0;

  setInterval(function growUp() {
    // Колбэк ссылается на переменную `self`,
    // значением которой является ожидаемый объект.
    self.age++;
  }, 1000);
}
// решение 2 - стрелочная функция

function Person() {
  this.age = 0;

  setInterval(() => {
    this.age++; // |this| должным образом ссылается на объект Person
  }, 1000);
}

var p = new Person();
```
