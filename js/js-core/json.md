# JSON

Почему не toString() – так как свойства могут меняться, добавляться и удаляться JSON – JS Object Notation
JSON.stringify для преобразования объект в JSON

```js
let student = {
  name: "John",
  isAdmin: false,
  courses: ["html", "css", "js"],
  wife: null,
};

let json = JSON.stringify(student); // для получения JSON-форматированным или сериализованы объектом  alert( typeOf json); // string

alert(json);
```

Строки используют двойные кавычки, имена свойств заключаются в двойные кавычки

поддержка: Объекты(с вложенными объектами), массивы, строки, числа, логические значения, болевые значения,
null

пропускает: методы, символьные свойства, свойства содержащие undefined, циклические ссылки

## reviver

```js
let value = JSON.parse(str, [reviver]); //str – JSON для преобразования в объект
//reviver – необязательная функция для каждой пары

let numbers = "[0, 1, 2, 3]";
numbers = JSON.parse(numbers);
alert(numbers[1]);

let user =
  '{ "name": "John", "age": 30, "isAdmin": false, "friends": [1,2,3,4]}"';
user = JSON.parse(user);
alert(user.friends[1]); //1  JSON не поддерживает комментарии
```

```js
let str = `{"title": "Conference", "date": "2017-11-30T12:00:00.000Z"}`;
//При дессериализации возникнет ошибка  let mmetuo = JSON.parse(str);

alert(meetup.date.getDate()); // Error как метод мог знать, что ему передают не строку, в объект даты
//исправим

let meetup = JSON.parse(str, function reviver(key, value) {
  if (key == "date") return new Date(value);
  return value;
});
alert(meetup.date.getDate()); //30  Работает и для вложенных методов
```

## JSON.stringify(obj)

Полный синтаксис JSON.stringify
let json = JSON.stringify(value[, replacer, space]); value – значение для кодирования
replacer – массив свойств для кодирования или функция function(key, value)
space – доп пространства используемые для форматирования

```js
// При передачи массива свойств будут закодированы только эти свойства
let room = {
  number: 23,
};

let meetup = {
  title: "Conference",
  participants: [{ name: "John" }, { name: "Alice" }],
  place: room,
}; //meetup ссылает на room
room.occupiedBy = meetup; //room ссылается на meetup  alert( JSON.stringify(meetup, ["title" , "participants"]) );
// {"title": "Conference", "participants": [{}, {}]} name нет в объекте так как мы их не задали

alert(
  JSON.stringify(meetup, ["title", "participants", "place", "name", number])
);
// {"title": "Conference", "participants": [{"name": "John"}, {"name": "Alice"}], "place" : {"number":  "23"} }

// список свойств очень большой
```

### replacer

```js
let room = { number: 23 };

let meetup = {
  title: "Conference",
  participants: [{ name: "John" }, { name: "Alice" }],
  place: room,
}; //meetup ссылает на room

room.occupiedBy = meetup;

alert(
  JSON.stringify(meetup, function replacer(key, value) {
    alert(`${key}: ${value}`);
    return key == "occupiedBy" ? undefined : value;
  })
);
```

<img src='./assets/js/json-replacer.png'>

функция replacer получает каждую пару ключ/значение, включая вложенные объекты. Применяет рекурсивно для
вложенных объектов. Значение this внутри replacer – это объект, который содержит текущее свойство

Первый вызов – особенный, ему придается специальный объект – обертка {"": meetup}, первая пара имеет пустой ключ, а значением является целевой объект поэтому первый вызов ":[Object Object]"

<!-- PROMISE ----------------------------------------------------------------------------------------------------------------------------->
