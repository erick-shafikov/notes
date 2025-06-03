# new Function

Синтаксис
let func = new Function([arg1, arg2, …argN], functionBody);

```js
new Function("a", "b", "return a + b");
new Function("a, b", "return a + b");
new Function("a, b", "return a + b");

let sum = new Function("a", "b", "return a+b");
alert(sum(1, 2));
```

Главное отличие, что функция создается из строки, на лету

Замыкание
функция запоминает, где родилась в свойстве Environment Это ссылка на внешнее лексическое окружение new function имеет доступ только к глобальным переменным

```js
function getFunc() {
  let value = test;
  let func = new Function("alert(value)");
  let func = function () {
    alert(value);
  };
  return func;
}

getFunc()(); // ошибка value не определенно | "test" из ЛО функции getFunc

// Если бы new Function имела доступ к внешним переменным, возникли бы проблемы с минификатором
```
