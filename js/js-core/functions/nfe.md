# NFE

!!!Не работает с FD

```js
let sayHi = function func(who) {
  alert(`Hello,${who}`);
};

// это позволяет ссылаться функции саму на себя и оно не доступно за пределами функции
let sayHi = function func(who) {
  if (who) {
    alert(`hello ${who}`);
  } else {
    func("guest");
  }
}; //так как FE

func(); // так работать не будет func не доступна вне функции

// Преимущество заключается в том, что sayHi может быть изменено

let sayHi = function (who) {
  if (who) {
    alert("Hello ${who}");
  } else {
    sayHi("Guest");
  }
};

let welcome = sayHi;
sayHi = null;
welcome(); //Не работает
```

старый вариант это arguments.callee. arguments.callee - вызов функции с ссылкой на саму себя

```js
var factorial = function (n) {
  return n == 1 ? 1 : n * arguments.callee(n - 1);
};

factorial(7);
```
