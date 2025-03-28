!!! join, slice, indexOf, push, splice методы обновляют свойство length

# arr.at()

возвращает элемент на указанной позиции, поддерживает отрицательные аргументы - отличие от обычного доступа

# arr.concat()

```js
arr.concat(arg1, arg2, arg3);
// arg1 arg2 – могут быть как массивами так и простыми значениям
```

- создаёт новый массив, в который копирует данные из других массивов и дополнительные значения.
- переданные ссылочные данные, копируются по ссылке

```js
let arr = [1, 2];
alert(arr.concat([3, 4])); // 1, 2, 3, 4

// взаимодействие с объектами

let arr = [1, 2];

let arrayLike = {
  0: "что-то",
  length: 1,
};

alert(arr.concat(arrayLike)); //[1, 2, [Object Object]]
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

```js
var alpha = ["a", "b", "c"];

var alphaNumeric = alpha.concat(1, [2, 3]); // Результат: ['a', 'b', 'c', 1, 2, 3]
```

# copyWithin()

Метод copyWithin() копирует последовательность элементов массива внутри него в позицию, начинающуюся по индексу target (первый аргумент). Копия берётся по индексам, задаваемым вторым и третьим аргументами start и end. Аргумент end является необязательным и по умолчанию равен длине массива.
TG

```js
const arr = [10, 20, 30, 40, 50];
const result = arr.copyWithin(0, -4, -2);
console.log(result); //[20, 30, 30, 40, 50]

[1, 2, 3, 4, 5].copyWithin(0, 3);
// [4, 5, 3, 4, 5]

[1, 2, 3, 4, 5].copyWithin(0, 3, 4);
// [4, 2, 3, 4, 5]

[1, 2, 3, 4, 5].copyWithin(0, -2, -1);
// [4, 2, 3, 4, 5]
```

# arr.entries()

вернет массив индекс - значение

```js
var arr = ["a", "b", "c"];
var eArr = arr.entries();

console.log(eArr.next().value); // [0, 'a']
console.log(eArr.next().value); // [1, 'b']
console.log(eArr.next().value); // [2, 'c']
```

# arr.every()

Метод every() проверяет, удовлетворяют ли все элементы массива условию, заданному в передаваемой функции.
метод возвращает true при любом условии для пустого массива.

arr.every(callback(currentValue[, index[, array]])[, thisArg])

```js
[12, 5, 8, 130, 44].every((elem) => elem >= 10); // false
[12, 54, 18, 130, 44].every((elem) => elem >= 10); // true
```

# arr.fill()

Заполняет массив значениям

arr.fill(value[, start = 0[, end = this.length]])

```js
[1, 2, 3].fill(4); // [4, 4, 4]
[1, 2, 3].fill(4, 1); // [1, 4, 4]
[1, 2, 3].fill(4, 1, 2); // [1, 4, 3]
[1, 2, 3].fill(4, 1, 1); // [1, 2, 3]
[1, 2, 3].fill(4, 3, 3); // [1, 2, 3]
[1, 2, 3].fill(4, -3, -2); // [4, 2, 3]
[1, 2, 3].fill(4, NaN, NaN); // [1, 2, 3]
[1, 2, 3].fill(4, 3, 5); // [1, 2, 3]
Array(3).fill(4); // [4, 4, 4]
[].fill.call({ length: 3 }, 4); // {0: 4, 1: 4, 2: 4, length: 3}

// Объекты заполняются по ссылке.
var arr = Array(3).fill({}); // [{}, {}, {}];
arr[0].hi = "hi"; // [{ hi: "hi" }, { hi: "hi" }, { hi: "hi" }]
```

# filter

Вернет массив, для всех проверок === true, иначе пустой массив

!!! При параллельном изменении - элементы не будут учтены, только при изменении в коллбеке функции

```js
filter((element) => {});
filter((element, index) => {});
filter((element, index, array) => {});
```

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

```js
// Изменение всех элементов
let words = ["spray", "limit", "exuberant", "destruction", "elite", "present"];

const modifiedWords = words.filter((word, index, arr) => {
  arr[index + 1] += " extra";
  return word.length < 6;
});

console.log(modifiedWords);
// Обратите внимание, что есть три слова длиной менее 6, но так как они были изменены,
// возвращается одно слово ['spray']

// Добавление новых элементов
words = ["spray", "limit", "exuberant", "destruction", "elite", "present"];
const appendedWords = words.filter((word, index, arr) => {
  arr.push("new");
  return word.length < 6;
});

console.log(appendedWords);
// Только три слова удовлетворяют условию, хотя `words` теперь имеет куда больше слов,
// длинной меньше 6 символов: ['spray', 'limit', 'elite']

// Удаление элементов
words = ["spray", "limit", "exuberant", "destruction", "elite", "present"];
const deleteWords = words.filter((word, index, arr) => {
  arr.pop();
  return word.length < 6;
});

console.log(deleteWords);
// Заметьте, что 'elite' не получено, так как удалено из `words` до того,
// как filter смог получить его: ['spray', 'limit']
```

# find (с остановкой)

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

# arr.findIndex()

возвращает индекс, на котором был найден элемент с заданными условиями иначе -1

arr.findIndex(callback[, thisArg])

# arr.findIndexLast()

ищет с конца

arr.findIndex(callback[, thisArg])

# arr.findLastIndex()

ищет с конца

# flat()

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

# flatMap()

позволяет применять функцию к каждому элементу и выпрямлять массив

```js
let arr1 = [1, 2, 3, 4];

arr1.map((x) => [x * 2]);
// [[2], [4], [6], [8]]

arr1.flatMap((x) => [x * 2]);
// [2, 4, 6, 8]

// выравнивается только один уровень
arr1.flatMap((x) => [[x * 2]]);
// [[2], [4], [6], [8]]
```

# forEach

позволяет запускать функцию для каждого элемента массива. Результат отбрасывается и игнорируется Его синтаксис:

Не существует способа остановить или прервать цикл forEach() кроме как выбрасыванием исключения. Если вам необходимо такое поведение, метод forEach() неправильный выбор.

```js
arr.forEach(function (item, index, array) {
  // ... делать что-то с item по умолчанию выполняет функции с this == undefined
});

[1, 2, 3, 4].forEach(alert);

[1, 2, 3, 4].forEach((item, index, array) => {
  alert(`${item} имеет позицию ${index} в ${array}`);
});
```

изменение массива во время вызова forEach

```js
let words = ["one", "two", "three", "four"];
words.forEach((word) => {
  console.log(word);
  if (word === "two") {
    words.shift();
  }
});
// one
// two
// four
```

# arr.includes

arr.includes(item, from)

ищет item, начиная с индекса from, и возвращает true, если поиск успешен. Используется, если не нужно знать индекс. Так же он правильно обрабатывает NaN Все методы используют строгое сравнение
при отрицательном from fromIndex + array.length

# arr.indexOf

arr.indexOf(item, from)

ищет item, начиная с индекса from, и возвращает индекс, на котором был найден искомый элемент, в противном случае -1.

# arr.lastIndexOf

arr.lastIndexOf(item, from) то же самое, но ищет справа налево.

# join

Он создаёт строку из элементов arr, вставляя glue между ними. Как работает:

- Пускай первым аргументом будет glue или, в случае отсутствия аргументов, им будет запятая ","
- Пускай result будет пустой строкой "".
- Добавить this[0] к result.
- Добавить glue и this[1].
- Добавить glue и this[2].
  …выполнять до тех пор, пока this.length элементов не будет склеено.
  ← result.

  !!! undefined и null будут преобразованы в пустую строку

```js
let array = [1, 0, false];

alert(arr.indexOf(0)); //1  alert( arr.indexOf(false) ); //2  alert( arr.indexOf(null) ); //-1
alert(arr.includes(1)); //true  const arr=[NaN];
alert(indexOf(NaN)); //-1, должно быть 0
alert(arr.includes(NaN)); //true
```

# arr.keys()

вернет итерируемый объект с индексами

# arr.lastIndexOf()

вернет найденный элемент начиная поиск от конца

# arr.map()

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

# arr.pop()

Удаляет последний элемент из массива и ← его:

# arr.push()

Добавляет элемент (или несколько элементов) в конец массива:

```js
fruits.push("Груша"); //←длину получившегося массива
```

# arr.reduce()

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

# arr.reduceRight()

работает аналогично, но только справа налево

```js
const arr = ["h", "e", "l", "l", "o"];
const result = arr.reduceRight((a, b) => b + a); //так к следующему добавляется предыдущий
console.log(result); //hello
```

# arr.reverse()

меняет порядок элементов в arr на обратный. Он также возвращает массив arr с изменённым порядком элементов.
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

# arr.shift()

Удаляет из массива первый элемент и ← этот элемент:

```js
fruits.shift();
```

# arr.slice()

возвращает новый массив, в который копирует элементы, начиная с индекса start и до end (не включая end). arr.slice([start], [end])

```js
let arr = ["t", "e", "s", "t"];

alert(arr.slice(1, 3)); // e, s копирует с 1 по 3 элементы
alert(arr.slice(-2)); // s, t копирует с -2 до конца

// Slice без аргументов создаёт копию массива, без изменения исходного
```

# arr.some()

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

# arr.sort()

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

# splice

удаляет, он начинает с позиции index, удаляет deleteCount элементов и вставляет elem1, ..., elemN на их место.
← массив из удалённых элементов, синтаксис: arr.splice(index[, deleteCount, elem1, ..., elemN])

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

# arr.toReversed()

в отличие от reverse() вернет массив, не меняя исходный

# arr.toSorted()

в отличие от sort() вернет массив, не меняя исходный

# arr.toSpliced()

в отличие от splice() вернет массив, не меняя исходный

# toString

Массивы по своему реализуют toString, который возвращает список элементов разделенных запятой
Массивы не имеют ни Symbol.toPrimitive ни valueOf, то есть с бинарным плюсом все действия происходят по правилу строки

так же есть метода toLocalString() который преобразует к данной локаль

# unshift

Добавляет элемент (или несколько элементов) в начало массива:

```js
fruits.unshift("Яблоко");
```

! Методы push и unshift могут добавлять сразу несколько элементов
! Массивы - это подвид объектов, при копировании ссылаются на один и тот же объект
! Варианты неправильного применения массивов – добавление нечислового свойства , создание дыр arr[0], а потом arr[100]
! Если массиву присваивается length меньше, чем текущее значение, то массив становится «короче»

# arr.values()

вернет итерируемый объект для перебора значений

# arr.with()

arr.with(index, value)

вернет новый массив, в котором элемент по индексу index будет заменен на value

- индекс может быть отрицательный
- RangeError при отсутствии индекса

# thisArg

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

# Symbol.iterator

arr[Symbol.iterator]()
