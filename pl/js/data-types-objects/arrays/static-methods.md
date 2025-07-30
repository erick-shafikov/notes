# Array.isArray(value)

Array.isArray(value) ← true, если value массив, и false, если нет.

Необязательный второй аргумент может быть функцией, которая будет применена к каждому элементу перед добавлением в массив, а thisArg позволяет установить this для этой функции.

```js
// range взят из примера выше,	возводим каждое число в квадрат
let arr = Array.from(range, (num) => num * num);
alert(arr); // 1,4,9,16,25
```

можно instanceof, но instanceof для 

```js
var iframe = document.createElement("iframe");
document.body.appendChild(iframe);
xArray = window.frames[window.frames.length - 1].Array;
var arr = new xArray(1, 2, 3); // [1,2,3]

// Correctly checking for Array
Array.isArray(arr); // true
// Considered harmful, because doesn't work through iframes
arr instanceof Array; // false

```

# Array.from()

принимает итерируемый объект или псевдо-массив и делает из него «настоящий» Array. После этого мы уже можем использовать методы массивов, thisArg позволяет установить this для этой функции: Array.from(obj[, mapFn, thisArg])
Пример:

```js
let arrayLike = { 0: "Hello", 1: "World", length: 2 };
let arr = Array.from(arrayLike); // (*)  alert(arr.pop()); // World (метод работает)
// Использование стрелочной функции в качестве функции отображения для

//второй аргумент - функция
// манипулирования элементами
Array.from([1, 2, 3], (x) => x + x);
// [2, 4, 6]

// Генерирования последовательности чисел, так как передается объект со свойством length итератор определяет это как array-like объект
Array.from({ length: 5 }, (v, k) => k); // v === undefined
// [0, 1, 2, 3, 4]
```

# Array.of()

вернет массив из аргументов переданных в функцию

```js
Array.of(7); // [7]
Array.of(1, 2, 3); // [1, 2, 3]

Array(7); // массив с 7 пустыми слотами
Array(1, 2, 3); // [1, 2, 3]

```