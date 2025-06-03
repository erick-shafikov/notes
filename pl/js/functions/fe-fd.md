# FE и FD

```js
function sayHi() {
  alert("hi");
}

let func = sayHi; // если бы было let func = sayHi(), то мы бы присвоили func значение вызова sayHi()
func(); //hi
sayHi(); //hi
```

- FD – Объявление функции через ключевое слово function foo(){},
- FE – создание функции через присваивание некой переменной let foo = function(){}.
- FE создается тогда, когда выполнение доходит до нее,
- FD всплывает, то есть мы не можем обратиться к функции, объявленной с помощью FE раньше ее создания.
- Видимость FD ограничивается {}, этот нюанс можно обойти с помощью присвоения переменной объявленной вне блока когда, а внутри блока с помощью FE присвоить функцию
- Function declaration, если находятся внутри блоков {}, вне их не видны, обойти можно про использовании FE

```js
// При FE в конце кода должна стоять ;
let sayHi = function () {
  alert("Привет");
};
let func = sayHi;
```

```js
console.log(square); // square поднят со значением undefined.
console.log(square(5)); // TypeError: square is not a function
var square = function (n) {
  return n * n;
};
```
