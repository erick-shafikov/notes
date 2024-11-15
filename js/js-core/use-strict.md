# USE STRICT

"use strict" – можно поставить в начале любой функции

- строгий режим делает невозможным случайное создание глобальных переменных.
- заставляет присваивания, которые всё равно завершились бы неудачей, выбрасывать исключения
- попытки удалить неудаляемые свойства будут вызывать исключения
- запрещает установку свойств primitive значениям
- this === undefined
- все обращения к необъявленным переменным – ошибка, как внутри блоков

```js
"use strict";
function bike() {
  console.log(this.name);
}
var name = "ninja";
var obj1 = { name: "some_field", bike: bike };
var obj2 = { name: "site", bike: bike };
bike(); // без "use strict" ninja, с "use strict" скрипт падает Uncaught TypeError: Cannot read properties of undefined (reading "name")
obj1.bike(); // без и с "use strict" some_field
obj2.bike(); // без "use strict" site

// TG use strict и повторяющиеся аргументы функции
function someFunc(a, b, b, c) {} //не выполнится use strict проверяет аргументы
```
