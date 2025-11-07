# Создание

Создание создать объект Date с текущими датой и временем:

- если передать время - то рассматривается как timestamp
- если передать строку, то ожидает yyyy-mm-dd
- два параметра и больше - год, месяц...
- UTC - гринвич

```js
let now = new Date();
alert(now); //показывает текущие дату и время
// Создать объект Date с временем, равным количеству миллисекунд (тысячная доля  секунды), прошедших с 1 января 1970 года UTC+0.
new Date(milliseconds); // 0 соответствует 01.01.1970 UTC+0
let Jan01_1970 = new Date(0);
alert(Jan01_1970);
// теперь добавим 24 часа и получим 02.01.1970 UTC+0
let Jan02_1970 = new Date(24 * 3600 * 1000);
alert(Jan02_1970);

let date = new Date("2017-01-26"); //Если аргумент всего один, и это строка, то из неё «прочитывается» дата.
var birthday = new Date("December 17, 1995 03:24:00");
var birthday = new Date("1995-12-17T03:24:00");

// Время не указано, поэтому оно ставится в полночь по Гринвичу и меняется в соответствии с часовым поясом места выполнения кода
// Так что в результате можно получить Thu Jan 26 2017 11:00:00 GMT+1100 (восточно-австралийское время) или Wed Jan 25 2017 16:00:00 GMT-0800 (тихоокеанское время)

new Date(year, month, date, hours, minutes, seconds, ms);
// Создать объект Date с заданными компонентами в местном часовом поясе. Обязательны только первые два аргумента.
// year должен состоять из четырёх цифр: значение 2013 корректно, 98 – нет.
// month начинается с 0 (январь)  по 11 (декабрь).
// Параметр date здесь представляет собой день месяца. Если параметр не задан, то принимается значение 1.
// Если параметры hours/minutes/seconds/ms отсутствуют, их значением становится 0.
// Например:
new Date(2011, 0, 1, 0, 0, 0, 0); // // 1 Jan 2011, 00:00:00
new Date(2011, 0, 1); // аналогично
Date(); // Wed Mar 12 2025 23:57:49 GMT+0500...
```

# статические методы

- Date.now() - timestamp
- Date.parse() - разбирает строку с датой в timestamp

Метод Date.parse(str) считывает дату из строки и ←таймстамп

- Формат строки должен быть следующим: YYYY-MM-DDTHH:mm:ss.sssZ, где:
- YYYY-MM-DD – это дата: год-месяц-день.
  Символ "T" используется в качестве разделителя.
  HH:mm:ss.sss – время: часы, минуты, секунды и миллисекунды.
  Необязательная часть "Z" обозначает часовой пояс в формате +-hh:mm. Если указать просто букву Z, то получим UTC+0.

```js
let ms = Date.pasre("2012-01-26T13:51:50.417-07:00");
alert(ms);

// Создать объект new Date из тайм стампа

let date = new Date(Data.parse("2012-01-26"));
alert(date);
```

- Date.UTC() - вернет время с учетом utc

# Методы экземпляра

## получение частей даты

- getFullYear() Получить год (4 цифры) вместе getYear() - проблема 2000
- getMonth() Получить месяц, от 0 до 11.
- getDate() Получить день месяца, от 1 до 31, что несколько противоречит названию метода.
- getDay() Вернуть день недели от 0 (воскресенье) до 6 (суббота)
- getHours(), getMinutes(), getSeconds(), getMilliseconds() Получить, соответственно, часы, минуты, секунды или миллисекунды.
- getTime() Для заданной даты возвращает таймстамп – количество миллисекунд, прошедших с 1 января 1970 года UTC+0.
- getTimezoneOffset() Возвращает разницу в минутах между местным часовым поясом и UTC:

```js
// текущая дата
let date = new Date();
// час в вашем текущем часовом поясе
alert(date.getHours());
// час в часовом поясе UTC+0 (лондонское время без перехода на летнее время)
alert(date.getUTCHours());
```

Помимо вышеупомянутых существуют utc аналоги getUTCHours, getUTCTime...

## приведение к строке

```js
//toString
date.toDateString(); //Fri Nov 07 2025
date.toISOString(); //2025-11-07T10:05:09.957Z
date.toJSON(); //2025-11-07T10:05:09.957Z
date.toLocaleDateString(); //07.11.2025
date.toLocaleString(); //07.11.2025, 15:05:09
date.toLocaleTimeString(); //15:05:09
date.toString(); //Fri Nov 07 2025 15:05:09 GMT+0500 (Узбекистан, стандартное время)
date.toTimeString(); //15:05:09 GMT+0500 (Узбекистан, стандартное время)
date.toUTCString(); //Fri, 07 Nov 2025 10:05:09 GMT

//toLocaleDateString
date.toLocaleDateString("en-US"); // "12/19/2012"
date.toLocaleDateString("en-GB"); // "20/12/2012"
date.toLocaleDateString("ko-KR"); // "2012. 12. 20."
date.toLocaleDateString("ar-EG"); // "٢٠/١٢/٢٠١٢"
date.toLocaleDateString("ja-JP-u-ca-japanese"); // "24/12/20"
date.toLocaleDateString(["ban", "id"]); // "20/12/2012"
```

## Установка компонента даты

Следующие методы позволяют установить компоненты даты и времени:

- setFullYear(year, [month], [date]) setMonth(month, [date]) setDate(date)
- setHours(hour, [min], [sec], [ms])
- setMinutes(min, [sec], [ms]) setSeconds(sec, [ms]) setMilliseconds(ms)
- setTime(milliseconds) (устанавливает дату в виде целого количества миллисекунд, прошедших с 01.01.1970 UTC)

```js
let today = new Date();
today.setHours(0); // сегодняшняя дата, но значение часа будет 0

today.setHours(0, 0, 0, 0); // всё ещё выводится сегодняшняя дата, но время будет ровно 00:00:00.

// Авто исправление даты
let date = new Date(2013, 0, 32); // 32 jan 2013 -> 1st feb 2013

// Неправильные компоненты даты автоматически определяются
let date = new Date(2016, 1, 28);
date.setDate(data.getDate() + 2); // 1 mar 2016

// Date.now() === Date.getTime()
```

```js
let start = Date.now();

for (let i=0; i<10000, i++){
  let doSomething = i*i*i
}
// let end= Date.now();
// alert(`цикл работал за ${start – end} миллисекунд`);
```

# Бенчмаркинг

```js
function diffSubtract(date1, date2) {
  // функция использует преобразование даты к строке
  return date2 - date1;
}

function diffGetTime(date1, date2) {
  // функция использует преобразование даты в объект
  return date2.getTime() - date1.getTime();
}

function bench(f) {
  let date1 = new Date(0);
  let date2 = new Date();

  let start = date.now();
  for (let i = 0; i < 1000; i++) {
    f(date1, date2);
  }

  return Date.now() - start;
}

alert(` Время diffSubtract:" + ${bench(diffSubtract)}+mс`);
alert(` Время diffSubtract:" + ${bench(diffGetTime)}+mс`);
```

# performance.now()

```js
performance.now(); // возвращает количество мс с начала загрузки сайта
```
