# Создание

```js
let arr = new Array(); // единственный аргумент – заданное число элементов в массиве, каждый из элементов undefined
let arr = new Array(2); //Создаёт массив длиной 2: [null, null]
let arr = new Array({}, {}, {}); //Создаёт массив: [{},{},{},{}]
// literal syntax
let arr = [];
//
let arr = Array.of(1, 2, 3);
let arr = Array
  .from
  // итерируемый объект
  ();

let arr = Array.from("string"); //[s, t, r, i, n, g]
```

В массиве могут храниться элементы любого типа:

```js
let arr = [
  " Яблоко",
  { name: "John" },
  true,
  function () {
    alert(" Привет ");
  },
];
```

# Деструктуризация массива

```js
let arr = ["Ilya", "Kantor"];
let [firstName, surname] = arr; // firstName = arr[0], surname = [1]
alert(surname); //Kantor
alert(firstName); //Ilya
let [fName, name] = "Ilya Kantor".split(" ");

// Пропуск элементов
let [name, , title] = ["Julius", "Cesar", "Consul", "of Romanic Republic"];
alert(title); //Consul
// Работает с любым перебираемым объектом
let [a, b, c] = "abc";
let [one, two, three] = new Set([1, 2, 3]);
// Присваивает что угодно с левой стороны
let user = {};
[user.name, user.surname] = "Ilya Kantor".split(" ");
alert(user.name); // Ilya
// Цикл с .entries()  Для объекта:
let userObj = {
  name: "John",
  age: 30,
};
for (let [key, value] of Object.entries(userObj)) {
  alert(`${key}: ${value}`);
}
// Для коллекции(тоже самое):
let userMap = new Map();
user.set("name", "John");
user.set("age", "30");
for (let [key, value] of userMap) {
  alert(`${key}: ${value}`);
}
```

# Остаточные параметры

```js
let [name1, name2, ...rest] = ["Julius", "Cesar", "Consul", "Roman Republic"];
alert(name1);
alert(name2);
alert(rest[0]); //Consul так как rest является массивом
alert(rest[1]); //Roman Republic
alert(rest.length); //2
```

# Значения по умолчанию

```js
let [firstName, surname] = [];
alert(firstName); //undefined

// Указание значений по	умолчанию. Они могут быть сложнее или даже функциями
// Простые значения по умолчанию:
let [name = "Guest", surname = "Anonyms"] = ["Julius"];
alert(name); //Julius
alert(surname); //Anonyms

// Использование prompt для значений по умолчанию

let [name = prompt("name?"), surname = prompt("surname?")] = ["Julius"];
alert(name); //Julius
alert(surname); //результат prompt
```

# Array.from

принимает итерируемый объект или псевдо-массив и делает из него «настоящий» Array. После этого мы уже можем использовать методы массивов, thisArg позволяет установить this для этой функции: Array.from(obj[, mapFn, thisArg])
Пример:

```js
let arrayLike = { 0: "Hello", 1: "World", length: 2 };
let arr = Array.from(arrayLike); // (*)  alert(arr.pop()); // World (метод работает)
```

# Array.isArray(value)

Array.isArray(value) ← true, если value массив, и false, если нет.

Необязательный второй аргумент может быть функцией, которая будет применена к каждому элементу перед добавлением в массив, а thisArg позволяет установить this для этой функции.

```js
// range взят из примера выше,	возводим каждое число в квадрат
let arr = Array.from(range, (num) => num * num);
alert(arr); // 1,4,9,16,25
```

# Перебор элементов

```js
// Цикл:
let arr = ["Яблоко", "Апельсин", "Груша"];

for (let i = 0; i < arr.length; i++) {
  alert(arr[i]);
}

// С помощью of:
for (let fruit of fruits) {
  //медленнее чем цикл
  alert(fruit);
}

const array = [1, 2, 3];
array.namedKey = 4;
let result = 0;
for (const key in array) {
  result += key;
}
console.log(result); //0012namedKey  for in проходится по всем свойствам array. При этом key всегда является строкой. Поэтому происходит конкатенация строк.
```

# Методы

## Методы. Concat

создаёт новый массив, в который копирует данные из других массивов и дополнительные значения. Его синтаксис: arr.concat(arg1, arg2...). arg1 arg2 – могут быть как массивами так и простыми значениям

```js
let arr = [1, 2];
alert( arr.concat( [3, 4]) ); // 1, 2, 3, 4

взаимодействие с объектами  let arr = [1, 2];

let arrayLike = {
0: "что-то",
length: 1
};

alert( arr.concat(arrayLike) );//[1, 2, [Object Object]]

```

Но если есть свойство Symbol.isConcatSpreadable то он обрабатывается как массив, вместо него добавляются
числовые свойства. Для корректной работы должны быть числовые свойства и свойство length

```js
let arr = [1, 2];

let arrayLike = {
  0: "что-то",
  1: "еще",
  [Symbol.isConcatSpreadable]: true,
  length: 2,
};

alert(arr.concat(arrayLike)); // 1,2, что-то, еще
```

## Методы. copyWithin()

Метод copyWithin() копирует последовательность элементов массива внутри него в позицию, начинающуюся по индексу target (первый аргумент). Копия берётся по индексам, задаваемым вторым и третьим аргументами start и end. Аргумент end является необязательным и по умолчанию равен длине массива.
TG

```js
const arr = [10, 20, 30, 40, 50];
const result = arr.copyWithin(0, -4, -2);
console.log(result); //[20, 30, 30, 40, 50]
```

## Методы. every()

Метод every() проверяет, удовлетворяют ли все элементы массива условию, заданному в передаваемой функции.
метод возвращает true при любом условии для пустого массива.

```js
[12, 5, 8, 130, 44].every((elem) => elem >= 10); // false
[12, 54, 18, 130, 44].every((elem) => elem >= 10); // true
```

## Методы. some()

проверяет, удовлетворяет ли какой-либо элемент массива условию, заданному в передаваемой функции.

```js
const array = [1, 2, 3, 4, 5]; // checks whether an element is even
const even = (element) => element % 2 === 0;
console.log(array.some(even));
// expected output: true

const list = [
  [false, " ", 0],
  [null, " ", {}],
  [undefined, -Infinity, Infinity],
];
const result = list.every((values) => values.some((value) => value)); //true в каждом подмассиве, есть что-то true
```

## Методы. forEach

позволяет запускать функцию для каждого элемента массива. Результат отбрасывается и игнорируется Его синтаксис:

```js
arr.forEach(function (item, index, array) {
  // ... делать что-то с item по умолчанию выполняет функции с this == undefined
});

[1, 2, 3, 4].forEach(alert);

[1, 2, 3, 4].forEach((item, index, array) => {
  alert(`${item} имеет позицию ${index} в ${array}`);
});
```

## Методы. join

Он создаёт строку из элементов arr, вставляя glue между ними. Как работает:

- Пускай первым аргументом будет glue или, в случае отсутствия аргументов, им будет запятая ","
- Пускай result будет пустой строкой "".
- Добавить this[0] к result.
- Добавить glue и this[1].
- Добавить glue и this[2].
  …выполнять до тех пор, пока this.length элементов не будет склеено.
  ← result.

## Методы поиска. indexOf, lastIndexOf, includes

- **arr.indexOf(item, from)** ищет item, начиная с индекса from, и возвращает индекс, на котором был найден искомый элемент, в противном случае -1.
- **arr.lastIndexOf(item, from)** то же самое, но ищет справа налево.
- **arr.includes(item, from)** ищет item, начиная с индекса from, и возвращает true, если поиск успешен. Используется, если не нужно знать индекс. Так же он правильно обрабатывает NaN Все методы используют строгое сравнение

```js
let array = [1, 0, false];

alert(arr.indexOf(0)); //1  alert( arr.indexOf(false) ); //2  alert( arr.indexOf(null) ); //-1
alert(arr.includes(1)); //true  const arr=[NaN];
alert(indexOf(NaN)); //-1, должно быть 0
alert(arr.includes(NaN)); //true
```

## Методы. fill

**arr.fill(value, start, end)** – заполняет массив повторяющимися value начиная с индекса start и заканчивая end

## Методы. filter

```js
let results = arr.filter(function (item, index, array) {
  // если true - элемент добавляется к результату, и перебор продолжается
  // возвращается пустой массив в случае, если ничего не найдено
});

let users = [
  { id: 1, name: "Вася" },
  { id: 2, name: "Петя" },
  { id: 3, name: "Маша" },
];
let someUsers = users.filter((item) => item.id < 3);
alert(someUsers.length); //2
```

## Методы. find (с остановкой)

```js
let result = arr.find(function (item, index, array) {
  // если функция возвращает true - возвращается текущий элемент и перебор прерывается
  // если все итерации оказались ложными, возвращается undefined
});
```

Функция вызывается по очереди для каждого элемента массива:
item – очередной элемент. (Имя объекта, а не ключ) index – его индекс. array – сам массив.

```js
let users = [
  { id: 1, name: "Вася" },
  { id: 2, name: "Петя" },
  { id: 3, name: "Маша" },
];

let user = users.find((item) => item.id == 1); // {id: 1, name: "Вася"}  alert( user.name)
```

## Методы. findIndex.

делает тоже самое, но только возвращает индекс, на котором был найден элемент с заданными условиями

## Методы. flat()

Метод flat() создает новый массив со всеми элементами под-массива, объединенными в него рекурсивно до указанной глубины. Параметр depth указывает, насколько глубоко должна быть сведена структура вложенного массива. По умолчанию 1. Если ты не знаешь уровень глубины, можешь передать Infinity в метод flat(). Тогда сглаживание будет максимальным. Также этот метод убирает пустые слоты.

```js
let arr1 = [0, 1, 2, [3, 4]];
console.log(arr1.flat()); // [0, 1, 2, 3, 4];
let arr2 = [0, 1, 2, [[3, 4]]];
console.log(arr2.flat(2)); // [0, 1, 2, 3, 4]
let arr3 = [0, 1, 2, [[3, 4, 5, 6, [7, 8, [9, 10]]]]];
console.log(arr3.flat(Infinity)); // [0, 1, 2, 3, 4]
let arr4 = [1, 2, , 4, 5];
console.log(arr4.flat()); // [1, 2, 3, 4]
```

## Методы. map

вызывает функцию для каждого элемента массива ← массив результатов выполнения этой функции.

```js
// Синтаксис:
let result = arr.map(function (item, index, array) {
  // возвращается новое значение вместо элемента
});

// Например, здесь мы преобразуем каждый элемент в его длину:
let lengths = ["Bilbo", "Gandalf", "Nazgul"].map((item) => item.length);
alert(lengths); // 5,7,6
```

## Методы. pop

Удаляет последний элемент из массива и ← его:

```js
fruits.pop();
```

## Методы. push

Добавляет элемент (или несколько элементов) в конец массива:

```js
  fruits.push("Груша") ←длину получившегося массива
```

## Методы. shift

Удаляет из массива первый элемент и ← этот элемент:

```js
fruits.shift();
```

## Методы. unshift

Добавляет элемент (или несколько элементов) в начало массива:

```js
fruits.unshift("Яблоко");
```

! Методы push и unshift могут добавлять сразу несколько элементов
! Массивы - это подвид объектов, при копировании ссылаются на один и тот же объект
! Варианты неправильного применения массивов – добавление нечислового свойства , создание дыр arr[0], а потом arr[100]
! Если массиву присваивается length меньше, чем текущее значение, то массив становится «короче»

## Методы. reduce

arr.reduce(function(previousValue, item, index, array))

- item – очередной элемент массива,
- index – его индекс,
- array – сам массив.

Функция применяется по очереди ко всем элементам массива и «переносит» свой результат на следующий вызов.
Аргументы:
previousValue – результат предыдущего вызова этой функции, равен initial при первом вызове (если передан initial),

```js
let arr = [1, 2, 3, 4, 5];
let result = arr.reduce((sum, current) => sum + current, 0);
alert(result); //15
```

## Методы. reduceRight

работает аналогично, но только справа налево

```js
const arr = ["h", "e", "l", "l", "o"];
const result = arr.reduceRight((a, b) => b + a); //так к следующему добавляется предыдущий
console.log(result); //hello
```

## Методы. reverse

меняет порядок элементов в arr на обратный. Он также возвращает массив arr с изменённым порядком
элементов.
Например:

```js
let arr = [1, 2, 3, 4, 5];
arr.reverse();
alert(arr); // 5,4,3,2,1

// TG;
function func(str) {
  return str.split("").reverse().join(""); //разделить на символы и перевернуть все символы и объединить в массив // .split(" ").reverse().join(" ")//разделить по пробелам и еще раз перевернуть уже слова
}
console.log(func("The quick brown fox")); //"ehT kciuq nworb xof"
```

## Методы. sort

возвращает отсортированный массив, но обычно возвращаемое значение игнорируется, так как изменяется сам
arr:
Пример для функции сравнения двух величин:

```js
function compare(a, b) {
  if (a > b) return 1; // если первое значение больше второго
  if (a == b) return 0; // если равны
  if (a < b) return -1; // если первое значение меньше второго
}

// По умолчанию элементы сортируются как строки
// TG
let arr = [5, 3, 22, true];
arr.sort();
console.log(arr); //22, 3, 5, true сравнивает как строки
```

Меняет исходный массив, что бы избежать этого можно использовать toSort()

## Методы. Slice

возвращает новый массив, в который копирует элементы, начиная с индекса start и до end (не включая end). arr.slice([start], [end])

```js
let arr = ["t", "e", "s", "t"];

alert(arr.slice(1, 3)); // e, s копирует с 1 по 3 элементы
alert(arr.slice(-2)); // s, t копирует с -2 до конца

// Slice без аргументов создаёт копию массива, без изменения исходного
```

## Методы. Splice

удаляет, он начинает с позиции index, удаляет deleteCount элементов и вставляет elem1, ..., elemN на их место. ← массив из удалённых элементов, синтаксис: arr.splice(index[, deleteCount, elem1, ..., elemN])

```js
// удаление:
let arr = ["Я", "Изучаю", "JS"];
arr.splice(1, 1);
alert(arr); // ["Я", "JS"]

// возврат удаленных в виде массива:
let arr = [1, 2, 3, 4];
let removed = arr.splice(0, 2);
alert(removed); // 1, 2

// вставка элементов без удаления:
let arr = [1, 2, 3];
arr.splice(2, 0, a, b);
alert(arr); // 1, 2, a, b, 3

// Отрицательные индексы разрешены:
arr.splice(-1, 0, 3, 4);
//начиная с индекса -1 (перед последним элементом) удалить 0 элементов и вставить 3, 4;
```

Так как этот метод меняет можно использовать toSplice

## Методы. toString

Массивы по своему реализуют toString, который возвращает список элементов разделенных запятой
Массивы не имеют ни Symbol.toPrimitive ни valueOf, то есть с бинарным плюсом все действия происходят по правилу строки

## Методы. thisArg

thisArg – необязательный параметр. Полный синтаксис всех методов arr.find(func, thisArg), arr.filter(func, thisArg), arr.map(func, thisArg). Значение thisArg становится this для func

```js
let army = {
  minAge: 18,
  maxAge: 27,
  canJoin(user) {
    return user.age >= this.minAge && user.age < this.maxAge;
  },
};
let users = [{ age: 16 }, { age: 20 }, { age: 23 }, { age: 30 }];

let soldiers = users.filter(army.canJoin, army); // найти пользователей, для которых army.canJoin  возвращает true

alert(soldiers.length); // 2
alert(soldiers[0].age); // 20
alert(soldiers[1].age); // 23
```
