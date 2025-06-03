# USE STRICT

"use strict" – можно поставить в начале любой функции

- строгий режим делает невозможным случайное создание глобальных переменных.

```js
"use strict";
// Предполагая, что не существует глобальной переменной
mistypeVaraible = 17; // mistypedVaraible, эта строка выбросит ReferenceError
// из-за опечатки в имени переменной
```

- заставляет присваивания, которые всё равно завершились бы неудачей, выбрасывать исключения
- попытки удалить неудаляемые свойства будут вызывать исключения
- запрещает установку свойств primitive значениям undefined = 5
- this === undefined в глобальном контексте
- все обращения к необъявленным переменным – ошибка, как внутри блоков
- попытка создать объект с повторяющимися свойствами
- попытка создать аргументы с одинаковыми именами
- удаление из глобального this
- работа с восьмеричными свойствами только через нотацию 0о без просто 0 в начале
- добавить свойство примитиву
- запрет функции with
- ограничение eval
- нельзя переопределить arguments

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
