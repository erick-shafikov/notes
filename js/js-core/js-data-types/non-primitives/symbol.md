<!--SYMBOL---------------------------------------------------------------------------------------------------------------------------------------->

## SYMBOL

Symbol – тип данных в JS, который образует всегда уникальные сущности, то есть два созданных символа никогда не будут равны друг другу

```js
// Синтаксис:
let id = Symbol(); //создаем новый символ
let id = Symbol("id"); //Символ с описанием id, если даже метки совпадают – символы уникальны

// Вывод описания:
alert(id); //TypeError cannot convert a Symbol to a string
alert(id.toString()); //Symbol(id)
alert(id.description); //id

// Добавление к объекту:
// символы позволяют создавать скрытые свойства объекта
let user = {
  name: "Вася",
};
let id = Symbol("id");
user[id] = 1;
// ИЛИ;
let id = Symbol("id");
let user = {
  name: "Вася",
  [id]: 123, // просто "id: 123" не сработает
};
```

- !!! Символы всегда уникальны, даже если их имена совпадают
- !!! Символы игнорируются циклом for...in
- !!! Символы не преобразуются автоматически в строки
- !!! Object.assign() копирует и символьные свойства

```js
const symbol = Symbol("key");
const test = Symbol.keyFor(symbol);
console.log(test); //undefined  Метод Symbol.keyFor(sum) извлекает общий ключ символа из глобального реестра символов для него. Для символов, созданных с помощью Symbol(description), такого ключа нет. "key" - это просто описание переменной symbol, нужен вызов test.description
```

### Глобальные символы:

Можно создать 2 символа который будут одинаковые – глобальные символы

```js
let id = Symbol.for("id");
let idAgain = Symbol.for("id");
alert(id === idAgain); // true

// для вывода глобальных символов, для вывода используется реестр глобальных символов, если символ не  глобальный, то он вернет undefined, а description доступен для любых символов
let sym = Symbol.for("name");
let sym2 = Symbol.for("id");
alert(Symbol.keyFor(sym)); // name
alert(Symbol.keyFor(sym2)); // id
Symbol.for(key);
// ищет символ по имени
Symbol.keyFor(sym);
// принимает глобальный символ и возвращает его имя.
```

### Symbol.toPrimitive

```js
// К строке:
// вывод  alert(obj);
// используем объект в качестве имени свойства
anotherObj[obj] = 123;
// К числу:
// явное преобразование  let num = Number(obj);
// математическое (исключая бинарный оператор "+")  let n = +obj; // унарный плюс
let delta = date1 - date2;
// сравнения больше/меньше
let greater = user1 > user2;
```

true == 1, false = 0
Все объекты – true
Преобразование к числу через Number(), или +a
Квадратные скобки дают намного больше возможностей, чем запись через точку.
При сравнении объекты превращаются в примитивы
Если принимающий объект уже имеет свойство с таким именем, оно будет перезаписано
Мат операции всегда преобразуют объекты в число, все кроме бинарного плюса (сложение строк)

Существует три преобразования в объектах: string, number, default. Все объекты – true.

```js
obj[Symbol.toPrimitive] = function (hint) {
  //должен вернуть примитивное значение
  //hint равен или number или string
};
// Пример:
let user = {
  name: "John",
  money: 1000,

  [Symbol.toPrimitive](hint) {
    alert(`hint:${hint}`);
    return hint == string ? `name:${this.name}` : this.money;
  },
};
alert(user); //hint:string {name:"JOhn"}
alert(+user); //hint:number -> 1000
alert(user + 500); //hint:default -> 1500
```

старый метод преобразование к примитивам, который работает также как и Symbol.toPrimitive

toString -> valueOf для хинта string
valueOf -> toString в ином случае

```js
let user = {
  name: "John",
  age: 20,

  toString() {
    return `name: ${this.name}`;
  },

  valueOf() {
    return this.money;
  },
};

alert(user); //{name: "John"}
alert(+user); //{age: 20}
alert(user + 500); //1500
```

## Symbol.unscopables

используется для указания значения объекта, чьи собственные и наследуемые имена свойств исключаются из привязок связанного объекта оператора with

```js
var keys = [];

with (Array.prototype) {
  keys.push("что-то");
}

Object.keys(Array.prototype[Symbol.unscopables]);
// ["copyWithin", "entries", "fill", "find", "findIndex", "includes", "keys", "values"]

var obj = {
  foo: 1,
  bar: 2,
};

obj[Symbol.unscopables] = {
  foo: false,
  bar: true,
};

with (obj) {
  console.log(foo); // 1
  console.log(bar); // ReferenceError: bar is not defined
}
```
