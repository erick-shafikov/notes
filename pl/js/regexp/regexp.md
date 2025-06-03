# Создание

Создание в двух вариантах, но вызывается new Regex

```js
let regex = /REGEX/g; //используем когда знаем какое будет выражение
let regex = new Regex("REGEX", "g"); //можно создавать на лету
```

# Флаги

- i - вне зависимости регистра
- g - global все совпадения не только первое
- m - многострочный режим
- u - поддержка юникода
- y - поиск на конкретной позиции
- d - превращается в массив с доп информацией
- s - точка соответствует символу перевода строки \n

# str.match(regex)

Метод который проверяет строку

```js
let str = "Любо, братцы, любо!";
const result = str.match(/любо/gi); // [Любо, любо]

result.index; // 0 (позиция совпадения)
result.input; // Любо, братцы, любо! (исходная строка)
// либо вернет null
```

# str.replace

Метод позволяет заменить

```js
"We will, we will".replace(/we/i, "I"); // I will, we will
```

# Специальные символы для вставки

- $& - вставляет всё найденное совпадение
- $` - вставляет часть строки до совпадения
- $' - вставляет часть строки после совпадения
- $n - [если n это 1-2 значное число, вставляет содержимое n-й скобочной группы регулярного выражения](#скобочные-группы-и-replace)
- $<name> вставляет содержимое скобочной группы с именем name
- $$ вставляет символ "$"

```js
"Люблю HTML".replace(/HTML/, "$& и JavaScript"); // Люблю HTML и JavaScript
```

# regexp.test(str)

Проверяет есть ли хоть одно совпадение, возвращает true если есть

## Символьные классы

позволяют искать группы символов:

- \d - символ от 0-9
- \s - символы пробела, \t табуляции, \n новой строки \v, \f и \r
- \w - буквы лат алфавита и \_
- \D - любой кроме \d
- \S - любой кроме \s
- \W - любой кроме \w

```js
"Есть ли стандарт CSS4?".match(/CSS\d/); //CSS4
"I love HTML5!".match(/\s\w\w\w\w\d/); // ' HTML5'
"+7(903)-123-45-67".match(/\d/g).join(""); // 79031234567
"+7(903)-123-45-67".replace(/\D/g, ""); // 79031234567
```

. - любой символ, но не отсутствие символе, при добавлении флага s символ перевода на новую строку \n тоже будет учитываться

```js
"A\nB".match(/A.B/); //null (нет совпадения)
"A\nB".match(/A.B/s); // A\nB (совпадение!)
```

- \p{…} для суррогатных пар
- - \p{L} - буквы в нижнем регистре Ll, модификаторы Lm, заглавные буквы Lt, в верхнем регистре Lu, прочие Lo
- - \p{N} - числа десятичная цифра Nd, цифры обозначаемые буквами (римские) Nl, прочие No
- - Знаки пунктуации P
- - Отметки M (например, акценты)
- - Символы S
- - Разделители Z
- - Прочие C

```js
"число: xAF".match(/x\p{Hex_Digit}\p{Hex_Digit}/u); // xAF
```

- \b - граница слова

```js
"Hello, Java!".match(/\bJava\b/); // Java
"Hello, JavaScript!".match(/\bJava\b/); // null
```

# Якоря

- ^ - начало строки
- $ - конец строки

```js
/^Mary/.test("Mary had a little lamb"); //true
/snow$/.test("it's fleece was white as snow"); // true
/^\d\d:\d\d$/.test("12:34"); // true
```

Многострочный режим включается флагом m.

```js
const str = `1е место: Винни
2е место: Пятачок
3е место: Слонопотам`;

str.match(/^\d/gm); // 1, 2, 3 ,без флага m нашел бы только первое число 1

str.match(/\d$/gm); // 1, 2, 3

str.match(/\d\n/g); // 1\n,2\n при поиске переносов строк не найдет последнее число 3, которое подходит
```

# Экранирование

Специальные символы - [ ] \ ^ $ . | ? \* + ( ).

```js
"Глава 5.1".match(/\d\.\d/); // 5.1 (совпадение!)
"Глава 511".match(/\d\.\d/); // null ("\." - ищет обычную точку)
"function g()".match(/g\(\)/); // "g()"
"1\\2".match(/\\/); // '\'
"/".match(/\//); // '/'
"/".match(new RegExp("/")); // находит / при создании new RegExp ненужно экранировать /
```

При передаче строки в new RegExp нужно удваивать обратную косую черту: \\ для экранирования специальных символов, потому что строковые кавычки «съедят» одну черту.

# Наборы и диапазоны

[...] - искать любой символ из заданных

```js
"Топ хоп".match(/[тх]оп/gi); // "Топ", "хоп"
```

- [a-z] - Поиск диапазонов символов

```js
"Exception 0xAF".match(/x[0-9A-F][0-9A-F]/g); // xAF
```

- \d – то же самое, что и [0-9]
- \w – то же самое, что и [a-zA-Z0-9_],
- \s – то же самое, что и [\t\n\v\f\r ]

- [^...] - исключающие диапазоны

```js
"alice15@gmail.com".match(/[^\d\sA-Z]/gi); // @ и .
```

Спец символы используем без экранирования . + ( ),

- Тире - не надо экранировать в начале или в конце (где оно не задаёт диапазон).
- ^ нужно экранировать только в начале (где он означает исключение).
- ] - если в конце

```js
"𝒳".match(/[𝒳𝒴]/u); // 𝒳
"𝒴".match(/[𝒳-𝒵]/u); // 𝒴
```

# Квантификаторы

- {n} - количество

```js
"Мне 12345 лет".match(/\d{5}/); //  "12345" найди 5 цифр \d{5} === \d\d\d\d\d
```

- {n, m} - Диапазон

```js
"Мне не 12, а 1234 года".match(/\d{3,5}/); // "1234" найди от \d{3} до \d{5}
```

- \d{3,} - от трех и более

```js
"Мне не 12, а 345678 лет".match(/\d{3,}/); // "345678"
```

- \+ - один или более (без слэша, для md-файла)

```js
"+7(903)-123-45-67".match(/\d+/g); // 7,903,123,45,67
```

- ? - ноль или один {0,1}

```js
str.match(/colou?r/g) ); // color, colour
```

- \* - ноль или более {0,} (без слэша, для md-файла)

```js
"100 10 1".match(/\d0*/g); // 100, 10, 1
"100 10 1".match(/\d0+/g); // 100, 10
```

Примеры

```js
"0 1 12.345 7890".match(/\d+\.\d+/g); // 12.345
"<body> ... </body>".match(/<[a-z]+>/gi); // <body>
"<h1>Привет!</h1>".match(/<[a-z][a-z0-9]*>/gi) ); // <h1>
"<h1>Привет!</h1>".match(/<\/?[a-z][a-z0-9]*>/gi) ); // <h1>, </h1>
```

Поиск по умолчанию - жадный, что бы включить ленивый нужен '?' то есть \*?, +? и ??

```js
'a "witch" and her "broom" is one'.match(/".+"/g); // "witch" and her "broom"
'a "witch" and her "broom" is one'.match(/".+?"/g); // "witch","broom"
"123 456".match(/\d+ \d+?/); // 123 4
'...<a href="link1" class="doc">... <a href="link2" class="doc">...'.match(
  /<a href=".*" class="doc">/g
); // <a href="link1" class="doc">... <a href="link2" class="doc">

str.match(/<a href=".*?" class="doc">/g); // <a href="link1" class="wrong">... <p style="" class="doc">
```

# скобочные группы

(...) - скобочная группа

```js
"Gogogo now!".match(/(go)+/gi); // "Gogogo"
"site.com my.site.com".match(/(\w+\.)+\w+/g); // site.com,my.site.com
"my@mail.com @ his@site.com.uk".match(/[-.\w]+@([\w-]+\.)+[\w-]+/g); // my@mail.com, his@site.com.uk
```

При работе с match - На позиции 0 будет всё совпадение целиком, На позиции 1 – содержимое первой скобочной группы, На позиции 2 – содержимое второй скобочной группы.

Вложенные скобочные группы

```js
let result = '<span class="my">'.match(/<(([a-z]+)\s*([^>]*))>/);
result[0]; // <span class="my">
result[1]; // span class="my"
result[2]; // span
result[3]; // class="my"
```

```js
let match = "a".match(/a(z)?(c)?/);

match.length; // 3
match[0]; // a всё совпадение
match[1]; // undefined
match[2]; // undefined

let match2 = "ac".match(/a(z)?(c)?/);

match2.length; // 3
match2[0]; // ac (всё совпадение)
match2[1]; // undefined, потому что для (z)? ничего нет
match2[2]; // c
```

# matchAll

При поиске всех совпадений (флаг g) метод match не возвращает скобочные группы.

```js
let results = "<h1> <h2>".matchAll(/<(.*?)>/gi);

// results - не массив, а перебираемый объект
results; // [object RegExp String Iterator]
results[0]; // undefined (*)
results = Array.from(results); // превращаем в массив
results[0]; // <h1>,h1 (первый тег)
results[1]; // <h2>,h2 (второй тег)

for (let result of results) {
  alert(result);
  // первый вывод: <h1>,h1
  // второй: <h2>,h2
}

let [tag1, tag2] = results;

tag1[0]; // <h1>
tag1[1]; // h1
tag1.index; // 0
tag1.input; // <h1> <h2>
```

# Именованные группы

комбинация символов - (?<**someGroupName**>) создает именованную группу

```js
let dateRegexp = /(?<year>[0-9]{4})-(?<month>[0-9]{2})-(?<day>[0-9]{2})/;
let str = "2019-04-30";

let groups = str.match(dateRegexp).groups;

groups.year; // 2019
groups.month; // 04
groups.day; // 30

// если нужны все совпадения, то нужно использовать matchAll
let dateRegexp = /(?<year>[0-9]{4})-(?<month>[0-9]{2})-(?<day>[0-9]{2})/g;

let str = "2019-10-30 2020-01-01";

let results = str.matchAll(dateRegexp);

for (let result of results) {
  let { year, month, day } = result.groups;

  alert(`${day}.${month}.${year}`);
  // первый вывод: 30.10.2019
  // второй: 01.01.2020
}
```

# Скобочные группы и replace

При replace, замены происходят с помощью $n

```js
let str = "John Bull";
let regexp = /(\w+) (\w+)/;

str.replace(regexp, "$2, $1"); // Bull, John

let regexp = /(?<year>[0-9]{4})-(?<month>[0-9]{2})-(?<day>[0-9]{2})/g;
let str = "2019-10-30, 2020-01-01";
str.replace(regexp, "$<day>.$<month>.$<year>"); // 30.10.2019, 01.01.2020
```

Исключение из конечного результата ?:

```js
let str = "Gogogo John!";

// ?: исключает go из запоминания
let regexp = /(?:go)+ (\w+)/i;

let result = str.match(regexp);

alert(result[0]); // Gogogo John (полное совпадение)
alert(result[1]); // John
alert(result.length); // 2 (больше в массиве элементов нет) (?:go) проигнорирует
```

# Обратные ссылки \N и \k<имя>

Обратная ссылка по номеру: \N. Применение - скобочная группа может начаться с одного или другого символа и закрыться им

```js
let str = `He said: "She's the one!".`;
// Результат - не тот, который хотелось бы
str.match(/['"](.*?)['"]/g); // "She'

str.match(/(['"])(.*?)\1/g); // "She's the one!"
// Обратная ссылка по имени: \k<имя>
alert(str.match(/(?<quote>['"])(.*?)\k<quote>/g)); // "She's the one!"
```

# Альтерация

Позволяет задать более гибкое ИЛИ-условие

Квадратные скобки работают только с символами или наборами символов. Альтернация мощнее, она работает с любыми выражения

```js
let regexp = /html|css|java(script)?/gi;

let str = "Сначала появился язык Java, затем HTML, потом JavaScript";

str.match(regexp); // Java,HTML,JavaScript
```

Чтобы применить частично, нужно ИЛИ-части заключить в скобки ()

```js
let regexp = /([01]\d|2[0-3]):[0-5]\d/g;

"00:00 10:10 23:59 25:99 1:2".match(regexp); // 00:00,10:10,23:59
```

# Опережающие и ретроспективные проверки

Для проверки последовательности шаблонов

## X(?=Y) Опережающая проверка

найти X при условии, что за ним следует Y

```js
let str = "1 индейка стоит 30€";

// найти цифру/цифры, которые идут до символа €
str.match(/\d+(?=€)/); // 30, число 1 проигнорировано, так как за ним НЕ следует €
// можно объединяться
// найти цифру/цифры, которые до пробела, до любых символов и 30
str.match(/\d+(?=\s)(?=.*30)/); // 1
```

## Негативная опережающая проверка X(?!Y)

найди такой X, за которым НЕ следует Y.

```js
let str = "2 индейки стоят 60€";

str.match(/\d+(?!€)/); // 2 (в этот раз проигнорирована цена)
```

# Ретроспективная проверка

Позитивная ретроспективная проверка: (?<=Y)X, ищет совпадение с X при условии, что перед ним ЕСТЬ Y.
Негативная ретроспективная проверка: (?<!Y)X, ищет совпадение с X при условии, что перед ним НЕТ Y.

```js
let str = "1 индейка стоит $30";

str.match(/(?<=\$)\d+/); // 30, одинокое число игнорируется
str.match(/(?<!\$)\d+/); // 2 (проигнорировалась цена)

//Скобочные группы
let str = "1 индейка стоит 30€";
let regexp = /\d+(?=(€|kr))/; // добавлены дополнительные скобки вокруг €|kr

str.match(regexp); // 30, €

let regexp = /(?<=(\$|£))\d+/;

str.match(regexp); // 30, $
```

# Флаг y поиск на заданной позиции

regexp.exec возвращают совпадения по очереди альтернатива str.matchAll

```js
let str = "let varName";

let regexp = /\w+/g;
alert(regexp.lastIndex); // 0 (при создании lastIndex=0)

let word1 = regexp.exec(str);
alert(word1[0]); // let (первое слово)
alert(regexp.lastIndex); // 3 (позиция за первым совпадением)

let word2 = regexp.exec(str);
alert(word2[0]); // varName (второе слово)
alert(regexp.lastIndex); // 11 (позиция за вторым совпадением)

let word3 = regexp.exec(str);
alert(word3); // null (больше совпадений нет)
alert(regexp.lastIndex); // 0 (сбрасывается по ок

while ((result = regexp.exec(str))) {
  alert(`Найдено ${result[0]} на позиции ${result.index}`);
  // Найдено let на позиции 0, затем
  // Найдено varName на позиции 4
}
```

Установка lastIndex

```js
let str = 'let varName = "value"';

let regexp = /\w+/g; // без флага g свойство lastIndex игнорируется

regexp.lastIndex = 4;

regexp.exec(str); // varName
```

Флаг y заставляет regexp.exec искать ровно на позиции lastIndex, ни до и ни после.

```js
let str = 'let varName = "value"';

let regexp = /\w+/y;

regexp.lastIndex = 3;
regexp.exec(str); // null (на позиции 3 пробел, а не слово)

regexp.lastIndex = 4;
regexp.exec(str); // varName (слово на позиции 4)
```

# Методы

## str.match(regexp)

разница есть ли флаг g ищет совпадения или возвращает null

```js
let str = "I love JavaScript";

// без g флага
let result = str.match(/Java(Script)/);
result[0]; // JavaScript (всё совпадение)
result[1]; // Script (первые скобки)
result.length; // 2
// Дополнительная информация:
result.index; // 7 (позиция совпадения)
result.input; // I love JavaScript (исходная строка)

// с g - флагом
let result = str.match(/Java(Script)/g);
result[0]; // JavaScript
result.length; // 1
```

## str.matchAll(regexp)

Он используется, в первую очередь, для поиска всех совпадений вместе со скобочными группами. возвращает не массив, а перебираемый объект

```js
let str = "<h1>Hello, world!</h1>";
let regexp = /<(.*?)>/g;

let matchAll = str.matchAll(regexp); // matchAll [object RegExp String Iterator], не массив, а перебираемый объект

matchAll = Array.from(matchAll); // теперь массив

let firstMatch = matchAll[0];
alert(firstMatch[0]); // <h1>
alert(firstMatch[1]); // h1
alert(firstMatch.index); // 0
alert(firstMatch.input); // <h1>Hello, world!</h1>
```

## str.split(regexp|substr, limit)

Разбивает строку в массив по разделителю – регулярному выражению regexp или подстроке substr

```js
"12-34-56".split("-"); // массив [12, 34, 56]
"12, 34, 56".split(/,\s*/); // массив [12, 34, 56]
```

## str.search(regexp)

возвращает позицию первого совпадения с regexp в строке str или -1

```js
"Я люблю JavaScript!".search(/Java.+/); // 8
```

## str.replace(str|regexp, str|func)

Когда первый аргумент replace является строкой, он заменяет только первое совпадение.

```js
"12-34-56".replace("-", ":"); // 12:34-56
"12-34-56".replace(/-/g, ":"); // 12:34:56
```

[Можно использовать специальный символы для вставки](#специальные-символы-для-вставки)

```js
"John Smith".replace(/(\w+) (\w+)/i, "$2, $1"); // Smith, John
```

для более сложных замен использует функция в качестве второго аргумента (аргумент func)

func(match, p1, p2, ..., pn, offset, input, groups):

- match
- p1, p2, ..., pn – содержимое скобок
- offset – позиция, на которой найдено совпадение
- input – исходная строка,
- groups – объект с содержимым именованных скобок

```js
"html and css".replace(/html|css/gi, (str) => str.toUpperCase()); //HTML and CSS
"Хо-Хо-хо".replace(/хо/gi, (match, offset) => offset); // 0-3-6
"John Smith".replace(
  /(\w+) (\w+)/,
  (match, name, surname) => `${surname}, ${name}`
); // Smith, John
// с остаточными аргументами
"John Smith".replace(/(\w+) (\w+)/, (...match) => `${match[2]}, ${match[1]}`); // Smith, John
// с именованными группами
"John Smith".replace(/(?<name>\w+) (?<surname>\w+)/, (...match) => {
  let groups = match.pop();

  return `${groups.surname}, ${groups.name}`;
}); // Smith, John
```

## regexp.exec(str)

без g флага ведет себя как str.match(regexp).

## regexp.test(str)

вернет true/false
