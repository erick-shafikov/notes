# ДАТА И ВРЕМЯ

Создание создать объект Date с текущими датой и временем:

```js
let now = new Date();
alert(now); //показывает текущие дату и время
// Создать объект Date с временем, равным количеству миллисекунд (тысячная доля  секунды), прошедших с 1 января 1970 года UTC+0.
new Date(milliseconds);
// 0 соответствует 01.01.1970 UTC+0
let Jan01_1970 = new Date(0);
alert(Jan01_1970);
// теперь добавим 24 часа и получим 02.01.1970 UTC+0
let Jan02_1970 = new Date(24 * 3600 * 1000);
alert(Jan02_1970);

new Date(dateString); //Если аргумент всего один, и это строка, то из неё «прочитывается» дата.
let date = new Date("2017-01-26");
alert(date);
// Время не указано, поэтому оно ставится в полночь по Гринвичу и
// меняется в соответствии с часовым поясом места выполнения кода
// Так что в результате можно получить
// Thu Jan 26 2017 11:00:00 GMT+1100 (восточно-австралийское время)
// или
// Wed Jan 25 2017 16:00:00 GMT-0800 (тихоокеанское время)

new Date(year, month, date, hours, minutes, seconds, ms);
// Создать объект Date с заданными компонентами в местном часовом поясе. Обязательны только первые два аргумента.
// year должен состоять из четырёх цифр: значение 2013 корректно, 98 – нет. month начинается с 0 (январь)  по 11 (декабрь).
// Параметр date здесь представляет собой день месяца. Если параметр не задан, то принимается значение 1.  Если параметры hours/minutes/seconds/ms отсутствуют, их значением становится 0.
// Например:
new Date(2011, 0, 1, 0, 0, 0, 0); // // 1 Jan 2011, 00:00:00
new Date(2011, 0, 1); // то же сам
```

### Получение компонента даты

- getFullYear() Получить год (4 цифры) getMonth() Получить месяц, от 0 до 11.
- getDate() Получить день месяца, от 1 до 31, что несколько противоречит названию метода.
  getHours(), getMinutes(), getSeconds(), getMilliseconds() Получить, соответственно, часы, минуты, секунды или миллисекунды.
- getDay() Вернуть день недели от 0 (воскресенье) до 6 (суббота)
- getTime() Для заданной даты возвращает таймстамп – количество миллисекунд, прошедших с 1 января 1970 года UTC+0.
- getTimezoneOffset() Возвращает разницу в минутах между местным часовым поясом и UTC:

```js
// текущая дата
let date = new Date();
// час в вашем текущем часовом поясе
alert(date.getHours());
// час в часовом поясе UTC+0 (лондонское время без перехода на летнее время) alert( date.getUTCHours() );
```

### Установка компонента даты

Следующие методы позволяют установить компоненты даты и времени:

- setFullYear(year, [month], [date]) setMonth(month, [date]) setDate(date)
- setHours(hour, [min], [sec], [ms])
- setMinutes(min, [sec], [ms]) setSeconds(sec, [ms]) setMilliseconds(ms)
- setTime(milliseconds) (устанавливает дату в виде целого количества миллисекунд, прошедших с 01.01.1970 UTC)

```js
let today = new Date();
today.setHours(0);
alert(today); // выводится сегодняшняя дата, но значение часа будет 0
today.setHours(0, 0, 0, 0);
alert(today); // всё ещё выводится сегодняшняя дата, но время будет ровно 00:00:00.
// Авто исправление даты
let date = new Date(2013, 0, 32); // 32 jan 2013
alert(date); // 1st feb 2013
// Неправильные компоненты даты автоматически определяются  let date = new Date(2016, 1, 28);
date.setDate(data.getDate()+2);
alert(date);// 1 mar 2016
// Date.now()
// Эквивалентен new Date.getTime()
let start = Date.now();
for (let i=0; i<10000, i++){
  let doSomething = i*i*i;
}
let end= Date.now();
alert(`цикл работал за ${start – end} миллисекунд`);

```

### Бенчмаркинг

```js
function diffSubtract(date1, date2){// функция использует преобразование даты к строке
return date2 – date1;
}

function diffGetTime(date1, date2){ // функция использует преобразование даты в объект
return date2.getTime() – date1.getTime();
}

function bench(f){
let date1 = newDate(0);
let Date2 = newDate()

let start = date.now();
for(let i = 0; i < 1000; i++) {f(date1, date2)};

return Date.now() – start;
}

alert(` Время diffSubtract:" + ${bench(diffSubtract)}+mс`);
alert(` Время diffSubtract:" + ${bench(diffGetTime)}+mс`);

```

### Методы

#### parse

Метод Date.parse(str) считывает дату из строки и ←таймстамп

Формат строки должен быть следующим: YYYY-MM-DDTHH:mm:ss.sssZ, где:
YYYY-MM-DD – это дата: год-месяц-день.
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

### performance.now()

performance.now() – возвращает количество мс с начала загрузки сайта
