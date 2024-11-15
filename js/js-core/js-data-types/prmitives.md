JS – динамически типизируемый язык

Примитивные значения:
number – числа до BigInt
BigInt число больше 2^53-1
NaN – результат вычислительной ошибки alert("не число"/2)
infinity = 1/0
undefined – значение не было присвоено
null – ссылка на несуществующий объект
Symbol – символ
string- строка

object – ссылочный тип данных

Что бы определить тип
typeof x или typeof (x) ← строку тип х

Работа с примитивами, как с объектами осуществляется через объект обертку, который создается на момент работы с примитивом

```js
let str = "Привет":

alert( str.toUpperCase() ); //ПРИВЕТ

let str = "Hi";

str.test = 5;

alert(str.test); //undefined (без strict), Ошибка (strict)
// В момент обращения создается объект обертка
// В строгом режиме попытка изменения выдаст ошибку
// Без строго режима операция продолжиться, объект получит свойство test, но после этого оно удаляется

```

в момент обращения к свойству строки str создается специальный объект, который знает значение строки и имеет полезные методы такие как .toUpperCase()
Этот метод запускается и возвращает строку
Специальный метод удаляется, остается изменённый str

Конструкторы String/Number/Boolean предназначены только для внутреннего пользования, то есть через new Number(1) или new Boolean(false)

```js
alert( typeOf 0); //Number
alert (typeOf new Number(0)) // Object

let zero = Number(0);  if(zero) {
alert("Разве ноль имеет истинное значение");
};


```

null/undefined не имеют методов

# Boolean

# Numbers

два тип чисел - Smi 2\*\*30 - 1 и HeapNumber

Шестнадцатеричные, восьмеричные, двоичные 0xff = 255 (16)
0b11111 = 255 (2)
0b377= 255 (8)

## Арифметические действия

```js
a += b; // a = a+b
a -= b; // a = a-b
a *= 2; // a = a*2
```

```js
let counter = 1;
let a = counter++;
alert(a); //1
// оператор «,» - каждое выражение выполняется и возвращается последнее  let a = (1 + 2, 3 + 4);
alert(a); //7
```

## BigInt

BigInt число больше 2^53-1, const bigInt = 1234567890123456789012345678901234567890n

```js
Разряды 10
1e9 = 1000000000
1e-9 = 0.000000001

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
return min + Math.random() * (max - min); // случайное число от min до max

let rand = min – 0.5 + Math.random() * (max – min + 1) //Случайной целое число
Math.round(rand);
let rand = min + Math.random() * (max - min);
Math.round(rand);
}

```

## Numbers BP. Учет плавающей точки

если умножить 2.2 на 100, получается число: 220.00000000000003, лучше оборачивать Math.round()

# String

Строки - массив символов

Обратные – помимо функции вставки значения выражения, так же помогают растянуть на несколько строк
!!! Строки неизменяемы

```js
Let guestList = `Guest:
John
Pete
Mary
;
```

new String('xxx') возвращает объект, что бы достать строку нужно вызвать

```js
const objStr = new String();

String.prototype.isPrototypeOf(objStr); //достать строку из объекта-строки
objStr.toString(); //или
```

- str.at(-1) - получить последний символ из строки
- str.length Возвращает длину строки
- str[i] символ на позиции в str, если символа нет undefined
- str.charAt(pos) Получить символ, который занимает позицию pos. Если символа нет, то пустую строку
- Перебор for (let char of "string")
- toLowerCase(), toUpperCase() пример: alert( "Interface"[0].toLowerCase() ); // "i"
- str.IndexOf(substring, pos) ищет подстроку substring в строке str, начиная с позиции pos, и возвращает позицию, на которой располагается совпадение, либо -1 при отсутствии совпадений.
- str.lastIndexOf(substr, position), который ищет с конца строки к её началу.
- Includes ← true, если в строке str есть подстрока substr, либо false, если нет. str.includes(substr, pos)
- str.startWith() и str.endsWith() проверяют, соответственно, начинается ли и заканчивается ли строка определённой строкой:

## Методы строк

- str.slice(start [, end]) ← строки от start до (не включая) end.
- str.substring(start [, end]) ← часть строки между start и end. (можно задавать start больше чем end) str.substr(start [, length]) ← часть строки от start длины length.
- str.codePointAt(pos) ← код для символа, находящегося на позиции pos:
- String.fromCodePoint(code) Создаёт символ по его коду code
- str.trim() - удаляет пробелы с обоих концов строки. Пробелы — это все пробельные символы (пробел, табуляция, неразрывный пробел и т. д.) и все символы конца строки (LF, CR и т. д.). Обрати внимание, trim() удаляет пробелы только с краев.
- str.repeat() - создает новую строку, повторяя заданную строку несколько раз, и возвращает ее. repeat() вызывает RangeError, если количество повторений отрицательное, равно бесконечности или превышает максимальный размер строки. Если используем параметр 0, возвращается пустая строка. При использовании нецелого числа значение преобразуется в ближайшее целое число с округлением вниз.
- split разбивает строку на массив по заданному разделителю delim. есть необязательный второй числовой аргумент
- replace(). Синтаксис str.replcae(regexp|substr, newSubstr|function[,flags])
  ← новую строку с некоторыми или всеми сопоставляенияси с шаблоном, замененным на заменитель

```js
let names = "Вася, Петя, Маша";
let arr = names.split(", ");
for (let name of arr) {
  alert(`сообщения получат ${name} `);
}

// необязательный числовой аргумент
let names = "Вася, Петя, Маша, Саша".split(", ", 2); // arr == ["Вася", "Петя"]
```

## Тегерированные аргументы

```js
//При использовании тегированных шаблонных литералов
//первым аргументом всегда будет массив строковых значений.
//Оставшимися аргументами будут значения переданных выражений!
function getPersonInfo(one, two, three) {
  console.log(one);
  console.log(two);
  console.log(three);
}
const person = "Lydia";
const age = 21;
getPersonInfo`${person} is ${age} years old` // в one – массив из : // пустой строки, так как начинается с параметра, // потом is , // потом разделенный строкой years old // в two – оправляется person // в three - 21
``;
```

# Преобразование типов

```js
alert(value); // преобразует значение к строке. ← undefined

String(value); // чтобы преобразовать значение к строке:
let value = Number(value); // преобразовывает value в число
let value = Boolean(value); // преобразовывает все в boolean
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

Логические преобразования

```js
0, null, undefined, NaN, ""; //false
// все остальные true
```
