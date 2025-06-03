# Function object

```js
// Пользовательские свойства
function sayHi() {
  alert("Hi");
  //подсчет вызовов
  sayHi.counter++;
}

sayHi.counter = 0;

sayHi();
sayHi();
alert(`вызвана ${sayHi.counter} раза`); //функция счетчик, через свойство
```

```js
//counter с использованием свойств функций
function makeCounter() {
  function counter() {
    return counter.count++;
  }
  counter.count = 0;
  return counter;
}

let counter = makeCounter();
counter.count = 10; //в этом и заключается преимущество над обычной переменной. Мы можем получить свойство вне блока кода
console.log(counter.count); //10
```

```js
function makeCounter() {
  let count = 0;

  function increaser() {
    return count++;
  }

  increaser.set = (value) => (count = value); //упаковка методов в increaser в силу того, что они разделяют  одну область видимости переменных
  increaser.decrease = () => count--;

  //increaser, increaser.set, increaser.decrease
  return increaser;
}

let counter = makeCounter();
alert(counter());
alert(counter());

counter.set(10); // работает так как counter возвращает функцию increaser и counter.decrease();

alert(counter());
```

## Свойство name

```js
function sayHi() {
  alert("hi");
}
alert(sayHi.name); //SayHi;

let sayHi = function () {
  alert("Hi");
};
alert(sayHi.name);
```

```js
// Добавление пользовательских свойств с безымянными функциями и именными с возвратом
// - у них нет this

function wrap1() {
  return function () {
    wrap.userProp = "x"; // не получится добавить пользовательские свойства
  };

  function wrap1() {
    wrap.userProp = obj;

    function wrap2(...arg) {
      // подкинем args  wrap2.userProp =
    }

    return wrap2;
  }
}
```

## свойство length

```js
// Встроенное свойство length содержит количество параметров функции
function f1(a) {}
function f2(a, b) {}
function many(a, b, ...more) {}
alert(f1.length); //1  alert(f2.length);//2
alert(many.length); //2
```

## свойство displayName

```js
function doSomething() {}

alert(doSomething.displayName); // "undefined"

var popup = function (content) {
  alert(content);
};

popup.displayName = "Показать всплывающее окно";

alert(popup.displayName); // "Показать всплывающее окно"
```
