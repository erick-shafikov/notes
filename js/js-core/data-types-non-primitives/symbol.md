# symbol

Symbol – тип встроенный объект, который образует всегда уникальные сущности (примитивные), то есть два созданных символа никогда не будут равны друг другу

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

- Символы всегда уникальны, даже если их имена совпадают
- Символы игнорируются циклом for...in
- Символы не преобразуются автоматически в строки
- Object.assign() копирует и символьные свойства

```js
const symbol = Symbol("key");
const test = Symbol.keyFor(symbol);
console.log(test); //undefined
```

Метод Symbol.keyFor(sum) извлекает общий ключ символа из глобального реестра символов для него. Для символов, созданных с помощью Symbol(description), такого ключа нет. "key" - это просто описание переменной symbol, нужен вызов test.description

# глобальные символы:

Можно создать 2 одинаковых символа– глобальные символы

при вызове Symbol.for идет поиск символа, если его нет, то создается новый, дял данного ключа

Глобальный реестр символов — это список со следующей структурой записей и пустой при инициализации

- глобальные не подлежат сборке мусора
- нельзя использовать в weak-map-set

```js
//создает символ
let id = Symbol.for("id");
//вернет обратно
let idAgain = Symbol.for("id");
alert(id === idAgain); // true

// для вывода глобальных символов используется реестр глобальных символов, если символ не глобальный, то он вернет undefined, а description доступен для любых символов
let sym = Symbol.for("name");
let sym2 = Symbol.for("id");

//вернет ключ из реестра глобальных символов
alert(Symbol.keyFor(sym)); // name
alert(Symbol.keyFor(sym2)); // id
Symbol.for(key); // ищет символ по имени
Symbol.keyFor(sym); // принимает глобальный символ и возвращает его имя.
```

```js
const globalSym = Symbol.for("foo"); // Global symbol
Symbol.keyFor(globalSym); // Expected output: "foo"

const localSym = Symbol(); // Local symbol
Symbol.keyFor(localSym); // Expected output: undefined
Symbol.keyFor(Symbol.iterator); // Expected output: undefined
```

<!-- статические свойства -->

# статические свойства

если функция-конструктор имеет метод с именем Symbol.hasInstance, то его поведение будет реализовано с помощью оператора instanceof

## Symbol.asyncIterator

возвращает AsyncIterator для использования в for await of

```js
const delayedResponses = {
  delays: [500, 1300, 3500],

  wait(delay) {
    return new Promise((resolve) => {
      setTimeout(resolve, delay);
    });
  },

  async *[Symbol.asyncIterator]() {
    for (const delay of this.delays) {
      await this.wait(delay);
      yield `Delayed response for ${delay} milliseconds`;
    }
  },
};

(async () => {
  for await (const response of delayedResponses) {
    console.log(response);
  }
})();

// Expected output: "Delayed response for 500 milliseconds"
// Expected output: "Delayed response for 1300 milliseconds"
// Expected output: "Delayed response for 3500 milliseconds"

const myAsyncIterable = {
  async *[Symbol.asyncIterator]() {
    yield "hello";
    yield "async";
    yield "iteration!";
  },
};

(async () => {
  for await (const x of myAsyncIterable) {
    console.log(x);
  }
})();
// Logs:
// "hello"
// "async"
// "iteration!"
```

## Symbol.hasInstance

используется instanceof

```js
class MyArray {
  static [Symbol.hasInstance](instance) {
    return Array.isArray(instance);
  }
}
console.log([] instanceof MyArray); // true
```

## Symbol.isConcatSpreadable

может ли быть сведен к массиву

```js
var x = [1, 2, 3];

var fakeArray = {
  [Symbol.isConcatSpreadable]: true,
  length: 2,
  0: "hello",
  1: "world",
};

x.concat(fakeArray); // [1, 2, 3, "hello", "world"]
```

## Symbol.iterator

для for of

```js
var myIterable = {};
myIterable[Symbol.iterator] = function* () {
  yield 1;
  yield 2;
  yield 3;
};
[...myIterable]; // [1, 2, 3]
```

## Symbol.match

можно ли использовать объект в качестве регулярного выражения

## Symbol.matchAll

можно ли использовать для итерируемого

## Symbol.replace

для совпадающих строк и их замены

## Symbol.search

метод для индексов внутри строки

## Symbol.species

используется в конструкторе для создания подобных объектов

```js
class MyArray extends Array {
  // Перегружаем species для использования родительского конструктора Array
  static get [Symbol.species]() {
    return Array;
  }
}
var a = new MyArray(1, 2, 3);
var mapped = a.map((x) => x * x);

console.log(mapped instanceof MyArray); // false
console.log(mapped instanceof Array); // true
```

## Symbol.split

разбивает строку на индексы

```js
class Split1 {
  constructor(value) {
    this.value = value;
  }
  [Symbol.split](string) {
    const index = string.indexOf(this.value);
    return `${this.value}${string.substring(0, index)}/${string.substring(
      index + this.value.length
    )}`;
  }
}

console.log("foobar".split(new Split1("foo")));
// Expected output: "foo/bar"
```

## Symbol.toPrimitive

преобразование объекта в примитивное значение

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

- true == 1,
- false = 0
- Все объекты – true
- Преобразование к числу через Number(), или +a
- Квадратные скобки дают намного больше возможностей, чем запись через точку.
- При сравнении объекты превращаются в примитивы
- Если принимающий объект уже имеет свойство с таким именем, оно будет перезаписано
- Мат операции всегда преобразуют объекты в число, все кроме бинарного плюса (сложение строк)

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
    return hint == "string" ? `name:${this.name}` : this.money;
  },
};
alert(user); //hint:string {name:"John"}
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

## Symbol.toStringTag

используется для описание объекта в toString()

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
