Работа с примитивами, как с объектами осуществляется через объект обертку, который создается на момент работы с примитивом
в момент обращения к свойству строки str создается специальный объект, который знает значение строки и имеет полезные методы такие как .toUpperCase() Этот метод запускается и возвращает строку
Специальный метод удаляется, остается изменённый str

Конструкторы String/Number/Boolean предназначены только для внутреннего пользования, то есть через new Number(1) или new Boolean(false)

- Строки - массив символов
- Строки неизменяемы

```js
let str = "Привет":
str.test = 5;
//undefined (без strict), Ошибка (strict)
// В момент обращения создается объект обертка
// В строгом режиме попытка изменения выдаст ошибку
// Без строго режима операция продолжиться, объект получит свойство test, но после этого оно удаляется

```

```js
alert(value); // преобразует значение к строке. undefined
```

# Перебор

```js
for (let char of "string") {
  //
}
```

<!-- unicode-символы ------------------------------------------------------------------------------------------------------------------------->

# unicode-символы

```js
"\xA9"; // "©"
"\u00A9"; // "©"
```

# Сравнение строк

Intl.Collator позволяет отфильтровать в алфавитном порядке с учетом локали

```js
const names = ["Hochberg", "Hönigswald", "Holzman"];

const germanPhoneBook = new Intl.Collator("de-DE-u-co-phonebk");

// as if sorting ["Hochberg", "Hoenigswald", "Holzman"]:
console.log(names.sort(germanPhonebook.compare).join(", "));
// logs "Hochberg, Hönigswald, Holzman"
```

# new String()

```js
//возвращает объект, что бы достать строку нужно вызвать
const objStr = new String();
objStr.toString();

const s = new String("foo"); // Создание объекта
console.log(s); // Отобразится: { '0': 'f', '1': 'o', '2': 'o'}
typeof s; // Вернёт 'object'

const a = new String("Hello world"); // a === "Hello world" is false { 0:'H', .... 10: 'd' }
const b = String("Hello world"); // b === "Hello world" is true
a instanceof String; // is true
b instanceof String; // is false
typeof a; // "object"
typeof b; // "string"
```

разница в поведении с eval

```js
const s1 = "2 + 2"; // создаёт строковый примитив
const s2 = new String("2 + 2"); // создаёт объект String
console.log(eval(s1)); // выведет число 4
console.log(eval(s2)); // выведет строку '2 + 2'
```

преобразование в примитив с помощью valueOf

```js
console.log(eval(s2.valueOf())); // выведет число 4
```

# toString()

если вызвать

```js
// то число будет переведено в систему исчисления
number.toString(2);
```

<!-- Статические методы ---------------------------------------------------------------------------------------------------------------------->

# статические методы String

## String.fromCharCode()

создает строку из последовательностей unicode

```js
String.fromCharCode(65, 66, 67); // "ABC"
```

## String.fromCodePoint()

с помощью кодовых точек

```js
String.fromCodePoint(42); // "*"
String.fromCodePoint(65, 90); // "AZ"
String.fromCodePoint(0x404); // "\u0404"
String.fromCodePoint(0x2f804); // "\uD87E\uDC04"
String.fromCodePoint(194564); // "\uD87E\uDC04"
String.fromCodePoint(0x1d306, 0x61, 0x1d307); // "\uD834\uDF06a\uD834\uDF07"

String.fromCodePoint("_"); // RangeError
String.fromCodePoint(Infinity); // RangeError
String.fromCodePoint(-1); // RangeError
String.fromCodePoint(3.14); // RangeError
String.fromCodePoint(3e-2); // RangeError
String.fromCodePoint(NaN); // RangeError
```

## String.raw()

строка из сырой шаблонной строки для вывода строки как есть

```js
String.raw`Привет\n${2 + 3}!`; // 'Привет\n5!',
String.raw`Привет\u000A!`; // 'Привет\u000A!', а здесь мы получим символы

let name = "Боб";
String.raw`Привет\n${name}!`; // 'Привет\nБоб!', сработала подстановка.

// Обычно вам не нужно вызывать метод String.raw() как функцию,
// но никто не запрещает вам делать это:
String.raw({ raw: "тест" }, 0, 1, 2);
// 'т0е1с2т'
```

<!-- свойства экземпляра ---------------------------------------------------------------------------------------------------------------------------->

# свойства экземпляра строк str

## str.length

длина строки

<!-- Методы экземпляра строк ---------------------------------------------------------------------------------------------------------------------------->

# методы экземпляра строк str

- str.anchor() - для оборачивания строки в a
- str.big() - для оборачивания строки в big
- str.blink() - для оборачивания строки в blink
- str.bold() - для оборачивания строки в bold
- str.fixed() - для оборачивания строки в tt
- str.fontcolor(color) - оборачивает в тег font с атрибутом color
- str.fontsize(size) - оборачивает в тег font с атрибутом size
- str.italics() - оборачивает в тег i
- str.link(url) - обернет строку в тег link c параметром url
- str.small() - обернет строку в тег small
- str.strike() - обернет строку в тег strike
- str.sub() - обернет строку в тег sub
- str.sup() - обернет строку в тег sup

## str[i]

символ на позиции в str, если символа нет undefined

## str.at(index)

Возвращает элемент на позиции

```js
// последний
str.at(-1);
```

## str.charAt(index)

позволяет получить символ или букву

## str.charCodeAt(index)

кодировка в ascii

## str.codePointAt(index)

⇒ UTF-16 код

## str.concat(str1, str2)

объединяет строки

- операторы str1 + str2 и str1 += str2 более производительные

```js
const hello = "Привет, ";
console.log(hello.concat("Кевин", ", удачного дня."));

/* Привет, Кевин, удачного дня. */
```

## str.startWith() и str.endsWith()

проверяют, соответственно, начинается ли и заканчивается ли строка определённой строкой:

```js
const str = "Быть или не быть, вот в чём вопрос.";

console.log(str.endsWith("вопрос.")); // true
console.log(str.endsWith("быть")); // false
console.log(str.endsWith("быть", 16)); // true
```

## str.includes(substr)

true, если в строке str есть подстрока substr, либо false, если нет. str.includes(substr, pos) начиная с pos позиции

- substr может быть регулярным выражением

## indexOf(substring, pos) и lastIndexOf(substr, position)

ищет подстроку substring в строке str, начиная с позиции pos, и возвращает позицию, на которой располагается совпадение, либо -1 при отсутствии совпадений.

который ищет с конца строки к её началу.

```js
// подсчет количества вхождений
const str = "Быть или не быть, вот в чём вопрос.";
const count = 0;
const pos = str.indexOf("в");

while (pos !== -1) {
  count++;
  pos = str.indexOf("в", pos + 1);
}

console.log(count); // отобразит 3
```

## str.isWellFormed()

проверяет есть ли суррогатные пары ⇒ boolean

## str.localCompare()

```js
const a = "réservé"; // With accents, lowercase
const b = "RESERVE"; // No accents, uppercase

console.log(a.localeCompare(b));
// Expected output: 1
console.log(a.localeCompare(b, "en", { sensitivity: "base" }));
// Expected output: 0
```

## str.match(regexp)

возвращает массив сопоставлений или null

```js
const str = "Смотри главу 3.4.5.1 для дополнительной информации";
const re = /смотри (главу \d+(\.\d)*)/i;
const found = str.match(re);

console.log(found);

// выведет [ 'Смотри главу 3.4.5.1',
//           'главу 3.4.5.1',
//           '.1',
//           index: 0,
//           input: 'Смотри главу 3.4.5.1 для дополнительной информации' ]
```

## str.matchAll(regexp)

вернет итератор по всем результатам

```js
const regexp = /t(e)(st(\d?))/g;
const str = "test1test2";

const array = [...str.matchAll(regexp)];

console.log(array[0]); // Expected output: Array ["test1", "e", "st1", "1"]
console.log(array[1]); // Expected output: Array ["test2", "e", "st2", "2"]
```

## str.normalize()

⇒ нормализованную Unicode форму строки - значения объекта String, на котором вызывается.

## str.padEnd(targetLength, padString) и str.padStart(targetLength, padString)

```js
// padEnd()
console.log("Блины со сметаной".padEnd(25, ".")); // Результат: "Блины со сметаной........"
console.log("200".padEnd(5)); // Результат: "200  "

//padStart()
"abc".padStart(10); // "       abc"
"abc".padStart(10, "foo"); // "foofoofabc"
"abc".padStart(6, "123465"); // "123abc"
"abc".padStart(8, "0"); // "00000abc"
"abc".padStart(1); // "abc"
```

## str.repeat(count)

создает новую строку, повторяя заданную строку несколько раз, и возвращает ее. repeat() вызывает RangeError, если количество повторений отрицательное, равно бесконечности или превышает максимальный размер строки. Если используем параметр 0, возвращается пустая строка. При использовании нецелого числа значение преобразуется в ближайшее целое число с округлением вниз.

## str.replace(regexp|substr, newSubstr|function[,flags])

Синтаксис str.replace(regexp|substr, newSubstr|function[,flags]) ← новую строку с некоторыми или всеми сопоставлениями с шаблоном, замененным на заменитель

- $$ - Вставляет символ доллара «$».
- $& - Вставляет сопоставившуюся подстроку.
- $` - Вставляет часть строки, предшествующую сопоставившейся подстроке.
- $' - Вставляет часть строки, следующую за сопоставившейся подстрокой.
- $n (или $nn) - Символы n или nn являются десятичными цифрами, вставляет - - n-ную сопоставившуюся подгруппу из объекта RegExp в первом параметре.

```js
const re = /([А-ЯЁа-яё]+)\s([А-ЯЁа-яё]+)/;
const str = "Джон Смит";
const newstr = str.replace(re, "$2, $1");
console.log(newstr); // Смит, Джон
```

```js
function f2c(x) {
  function convert(str, p1, offset, s) {
    return ((p1 - 32) * 5) / 9 + "C";
  }
  const s = String(x);
  const test = /(\d+(?:\.\d*)?)F\b/g;
  return s.replace(test, convert);
}
```

## str.replaceAll()

- возвращает новую строку со всеми совпадениями

## str.search()

аналог test, но вместо подстроки возвращает ее индекс

## slice(start [, end])

строки от start до (не включая) end.

- в отличает от str.substring может принимать отрицательные индексы

```js
let str1 = "Приближается утро.";
str1.slice(1, 8); //риближа
str1.slice(4, -2); //лижается утр
str1.slice(12); //утро.
str1.slice(30); //""
str1.slice(-3); //вернёт 'ро.'
str1.slice(-3, -1); //вернёт 'ро'
str1.slice(0, -1); //'Приближается утро'
str.slice(-11, 16); //вернёт 'ается утр'
```

## str.split(delim)

разбивает строку на массив по заданному разделителю delim (символ или регулярное выражение). есть необязательный второй числовой аргумент - лимит на количество подстрок

```js
// необязательный числовой аргумент
let names = "Вася, Петя, Маша, Саша".split(", ", 2); // arr == ["Вася", "Петя"]
```

## substring(start [, end])

- не может принимать отрицательные индексы в отличие от slice

часть строки между start и end. (можно задавать start больше чем end) str.substr(start [, length]) ← часть строки от start длины length.

```js
const anyString = "Mozilla";

anyString.substring(0, 3); // Отобразит 'Moz'
anyString.substring(3, 0); // Отобразит 'Moz'
anyString.substring(4, 7); // Отобразит 'lla'
anyString.substring(7, 4); // Отобразит 'lla'
anyString.substring(0, 6); // Отобразит 'Mozill'
anyString.substring(0, 7); // Отобразит 'Mozilla'
anyString.substring(0, 10); // Отобразит 'Mozilla'
```

## str.toLowerCase(), str.toUpperCase(), str.toLocaleLowerCase(), str.toLocaleUpperCase(), str.toLowerCase()

Возвести в upper или lower case

## str.toWellFormed()

преобразует строку в well-formed

## str.trim(), str.trimRight(), str.trimLeft()

удаляет пробелы с обоих концов строки. Пробелы — это все пробельные символы (пробел, табуляция, неразрывный пробел и т. д.) и все символы конца строки (LF, CR и т. д.). Обрати внимание, trim() удаляет пробелы только с краев.

- trimRight - уберет символы справа
- trimLeft - уберет символы слева

## str.valueOf()

вернет примитивное значение для объекта String

## String.prototype[@@iterator]()

```js
const string = "A\uD835\uDC68";

const strIter = string[Symbol.iterator]();

console.log(strIter.next().value); // "A"
console.log(strIter.next().value); // "\uD835\uDC68"
```

<!-- шаблонные строки ------------------------------------------------------------------------------------------------------------------------>

# шаблонные строки

Обратные – помимо функции вставки значения выражения, так же помогают растянуть на несколько строк
Для многострочного текста

```js
Let guestList = `Guest:
John
Pete
Mary
;
```

```js
`\`` === "`"; //true
```

# Теговые аргументы

- первый аргумент - массив строчных значений
- остальные - аргументы для подстановок

```js
const person = "Mike";
const age = 28;

//strings - массив строк из литерала
function myTag(strings, personExp, ageExp) {
  const str0 = strings[0]; // "That "
  const str1 = strings[1]; // " is a "

  // Технически, в конце итогового выражения
  // (в нашем примере) есть ещё одна строка,
  // но она пустая (""), так что пропустим её.
  // const str2 = strings[2];

  let ageStr;
  if (ageExp > 99) {
    ageStr = "centenarian";
  } else {
    ageStr = "youngster";
  }

  // Мы даже можем вернуть строку, построенную другим шаблонным литералом
  return `${str0}${personExp}${str1}${ageStr}`;
}

const output = myTag`That ${person} is a ${age}`;

console.log(output);
// That Mike is a youngster
```

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
getPersonInfo`${person} is ${age} years old`;
// в one – массив из пустой строки, так как начинается с параметра, потом " is "", потом разделенный строкой " years old"
// массив из всех не-аргументов ['', ' is ', ' years old']
// в two – оправляется person Lydia
// в three age - 21
```

## raw

```js
function tag(strings) {
  return strings.raw[0];
}

tag`string text line 1 \\n string text line 2`;
// выводит "string text line 1 \\n string text line 2",
// включая 'n' и два символа '\'
```
