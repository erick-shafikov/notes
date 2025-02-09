# Объявления чисел

number – числа до BigInt

```js
let value = Number(value); // преобразовывает value в число
```

два тип чисел - Smi 2\*\*30 - 1 и HeapNumber

- 0, 117 и -345 десятичная система счисления
- 015, 0001 и -077 восьмеричная система счисления
- 0x1123, 0x00111 и -0xF1A7 шестнадцатеричная система счисления
- 0b11, 0b0011 и -0b11 двоичная система счисления

С плавающей точкой

- 3.14
- -3.1E+12
- -.3333333333333333333
- .1e-23

infinity = 1/0

## Арифметические действия

```js
a += b; // a = a + b
a -= b; // a = a - b
a *= 2; // a = a * 2
```

```js
let counter = 1;
let a = counter++;
alert(a); //1
// оператор «,» - каждое выражение выполняется и возвращается последнее
let a = (1 + 2, 3 + 4);
alert(a); //7
```

## BigInt

BigInt число больше 2^53 - 1, const bigInt = 1234567890123456789012345678901234567890n

```js
// Разряды 10
const x = 1e9; //= 1000000000
const y = 1e-9; //= 0.000000001
```

## isNan()

```js
const string = "string";
const number = 21;

// Number.isNan проверяет ЧИСЛО на isNan
console.log(Number.isNaN(name));//false не число
console.log(Number.isNaN(number));//false число, но не Nan

isNan проверяет переданное значение на Nan
console.log(isNaN(name));//true так как это не число
console.log(isNaN(number));//false так как это число, но не Nan
```

Проверить на NaN === value нельзя, т.к. NaN не равно NaN

## Методы чисел

- toString(base) Метод num.toString(base) возвращает строковое представление числа num в системе счисления base., ставить 2 точки, так как JS может подумать, что начинается десятичная дробь 123456..toString(36) // 2n9c
- Math.floor(num) Округление в меньшую сторону: 3.1 становится 3, а -1.1 — -2.
- Math.ceil(num) Округление в большую сторону: 3.1 становится 4, а -1.1 — -1.
- Math.round(num) Округление до ближайшего целого: 3.1 становится 3, 3.6 — 4, а -1.1 — -1.
- Math.trunc(num) Отбрасывает дробную часть без округления

```js
alert(String(Math.trunc(Number("49")))); // "49", то же самое ⇒ свойство целочисленное
alert(String(Math.trunc(Number("+49")))); // "49", не то же самое, что "+49" ⇒ свойство не  целочисленное
alert(String(Math.trunc(Number("1.2")))); // "1", не то же самое, что "1.2" ⇒ свойство не  целочисленное
```

- ToFixed(n) округляет число до n знаков после запятой и возвращает строковое представление результата.
- isFinite(value) преобразует аргумент в число и ←true, если оно является обычным числом, т.е. не NaN/Infinity/-Infinity:
- parseInt(mum, notation) ←целое число:

```js
alert(parseInt("100px")); // 100
```

- parseFloat(mum, notation) ← число с плавающей точкой:

```js
alert(parseFloat("12.5em")); // 12.5
```

- Math.random() ← псевдослучайное число в диапазоне от 0 (включительно) до 1 (но не включая 1)
- Math.max(a, b, c...) / Math.min(a, b, c...) ← наибольшее/наименьшее число из перечисленных аргументов.

```js
function applyCurrent(num) {
  Math.max(min, Math.min(max, num)); //функция для попадания в диапазон
}
```

- Math.pow(n, power) ← число n, возведённое в степень power

## Numbers BP. Random number

```js
// генерация случайного целого числа
function randomIntFromInterval(min, max) {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function random(min, max) {
return min + Math.random() * (max - min); }// случайное число от min до max

let rand = min – 0.5 + Math.random() * (max – min + 1) //Случайной целое число
Math.round(rand);
let rand = min + Math.random() * (max - min);
Math.round(rand);


```

## Числовые преобразования

```js
Number(undefined); //NaN
Number(null); //0;
Number(true); //1
Number(false); //0
Number(""); //0
Number("      "); //0
Number("some string"); //NaN
```

# конструктор new Number()

```js
alert(typeof 0); //Number
alert(typeof new Number(0)); // Object

let zero = Number(0);
if (zero) {
  alert("Разве ноль имеет истинное значение");
}
```

## Numbers BP. Учет плавающей точки

если умножить 2.2 на 100, получается число: 220.00000000000003, лучше оборачивать Math.round()
