<!-- Создание объекта ---------------------------------------------------------------------------------------------------------------------->

# Создание объекта

Объекты - это область памяти, ссылка или указатель

Варианты создания объектов

```js
// литеральный тип
const obj = {};
// через функции конструкторы
const obj = new ObjFunctionCreator(); // ObjFunctionCreator = function(){this. = ....}
// Object.create
Object.create(proto, [descriptors]);
// fromEntries
Object.fromEntries([
  ["key0", "value0"],
  ["key1", "value1"],
]);
```

Ключами могут быть и пустые строки и знак !

```js
// Создание
let user1 = new Object(); // синтаксис "конструктор объекта"
let user2 = {}; // синтаксис "литерал объекта"
let user3 = {
  // объект
  name: "John", // под ключом "name" хранится значение "John"
  age: 30, // под ключом "age" хранится значение 30
};
// Получение свойства объекта:
alert(user.name); // John
alert(user.age); // 30
// Удаление
delete user.age;

// Для ключа, состоящего из двух слов:
user["likes birds"] = true;
alert(user["likes birds"]); // true
delete user["likes birds"];

let key = "likes birds"; // то же самое, что и user["likes birds"] = true;
user[key] = true;
```

Вычисляемые свойства. Смысл вычисляемого свойства прост: запись [fruit] означает, что, имя свойства необходимо взять из переменной fruit. И если посетитель введёт слово "apple", то в объекте bag теперь будет лежать свойство {apple: 5}.

```js
let fruit = prompt("Какой фрукт купить?", "apple");
let bag = {
  [fruit]: 5, // имя свойства будет взято из переменной fruit
};

alert(bag.apple); // 5, если fruit="apple"
```

```js
let obj = {
  0: 1,
  0: 2,
};
console.log(obj["0"] + obj[0]); //4 так как при обращению с свойству мы ожидаем вычисляемое свойство

// TG;
const a = {};
const b = { key: "b" };
const c = { key: "c" };
a[b] = 123;
a[c] = 456;
console.log(a[b]); //456 ключи автоматически конвертируются в строки так что объект превращается в [object Object] в двух последних присваиваниях
```

Объекты константы:

```js
const user = { name: "John" };
user.age = 25; // (*)  alert(user.age); // 25
const user = { name: "John" };
// Ошибка (нельзя переопределять константу user)
user = {
  name: "Pete",
};
```

Короткие свойства:

```js
function makeUser(name, age) {
  return { name: name, age: age };
}
// Тоже самое:
function makeUser(name, age) {
  return {
    name,
    age,
  };
}
```

## Методы

```js

```

Проверка наличия свойства:

```js
// Проверка наличия свойства:
let user = { age: 30 };
let key = "age";
alert(key in user); // true
```

Перебор

```js
for (key in object) {
  //
}
// Для вывода и ключей и свойств:
for (let key in user) {
  alert(key); // выведет значения ключей
  alert(user[key]); //выведет свойства
}
```

Ссылка на объект находится в стеке, сам объект находится в куче

# Копирование объектов

```js
Object.assign(dest, [src1, src2, src3]);

// Для объединения:
let user = { name: "John" };
let permissions1 = { canView: true };
let permissions2 = { canEdit: true };
// копируем все свойства из permissions1 и permissions2 в user  Object.assign(user, permissions1, permissions2);
// now user = { name: "John", canView: true, canEdit: true }

// Для копирования объекта:
let user = { name: "John", age: 30 };
let clone = Object.assign({}, user);
// Объекты с не примитивными свойствами
let user = {
  name: "John",
  sizes: {
    height: 182,
    width: 50,
  },
};
alert(user.sizes.height); // 182

// Специальная функция js для копирования объектов
structureClone(obj); //нельзя скопировать функции, dom-элементы
```

# Дескрипторы

У каждого свойства есть три атрибута (флага):

- writable – если true, то свойство можно изменить
- enumerable – если true, то свойство можно перечислять в циклах
- configurable – если true, свойство можно удалить, а эти атрибуты можно изменять, иначе этого делать нельзя. При создании свойств, все флаги – true.

```js
// Чтобы получить полную информацию о свойстве
let descriptor = Object.getOwnPropertyDescriptor(obj, propertyName);
//где obj – Объект,
//propertyName – имя свойства,
// возвращаемый объект – дескриптор свойства
let user = {
  name: "John",
};
let descriptor = Object.getOwnPropertyDescriptor(user, "name");
alert(JSON.stringify(descriptor, null, 2));
// {"value":john,  "writable":true,  "enumerable":true ,  "configurable":"true"}

// Чтобы изменить
Object.defineProperty(obj, propertyName, descriptor); //obj, propertyName – объект и его свойство,
// descriptor – применяемый дескриптор. если свойство существует, о его флаги обновятся, если нет, то  метод создает новое свойство, если флаг не указан, то ему присваивается false.

let user = {};
Object.defineProperty(user, "name", { value: "John" }); // value: "John", все флаги –false
```

```js
// только для чтения:
let user = { name: "john" };

Object.defineProperty(user, "name", { writable: false });

user.name = "Pete"; // ошибка только в строгом режиме, изменить только новым вызовом Object.defineProperty
```

```js
// Не перечисляемое свойство
let user = {
  name: "John",
  toString() {
    return this.name;
  },
};
for (let key in user) alert(key); // name, toString

Object.defineProperty(user, "toString", {
  enumerable: false,
});
```

неконфигурируемое свойство

```js
let descriptor = Object.getOwnPropertyDescriptor(Math, "PI");
alert(Json.stringify(descriptor, null, 2));
// Определение свойсва, как не конфигурируемого – это дорога в один конец, его нельзя будет изменить
let user = {};
Object.defineProperty(user, "name", {
  value: "John",
  writable: false,
  configurable: false,
});
// теперь невозможно изменить user.name или его флаги
// всё это не будет работать:
//	user.name =  "Pete"
//	delete user.name
//	defineProperty(user, "name", ...)
Object.defineProperty(user, "name", { writable: true }); // Ошибка
```

```js
Object.defineProperties(obj, {
  // позволяет определить и расставить флаги для нескольких свойств
  prop1: descriptor1,
  prop2: descriptor2,
});

Object.defineProperties(user, {
  name: { value: "John", writable: false },
  surname: { value: "Smith", writable: false },
  // ...
});
```

Чтобы получить все дескрипторы свойств сразу, можно воспользоваться методом Object.getOwnPropertyDescriptors(obj).

```js
let descriptor = Object.getOwnPropertyDescriptor(obj, propertyName);
```

Вместе с Object.defineProperties этот метод можно использовать для клонирования объекта вместе с его флагами:

```js
let clone = Object.defineProperties({}, Object.getOwnPropertyDescriptors(obj));
// Обычно при клонировании объекта мы используем присваивание, чтобы скопировать его свойства:

for (let key in user) {
  clone[key] = user[key];
}
```

…Но это не копирует флаги. Так что если нам нужен клон «получше», предпочтительнее использовать
Object.defineProperties.

Другое отличие в том, что for..in игнорирует символьные свойства, а Object.getOwnPropertyDescriptors возвращает дескрипторы всех свойств, включая свойства-символы.

Методы запечатывания объекта:

```js
Object.preventExtensions(obj); //запрещает добавлять новые свойства в объект
Object.seal(obj); //запрещает добавлять и удалять
Object.freeze(obj); //запрещает добавлять и удалять и изменять
// проверки на ограничения
Object.isExtensible(obj);
Object.isSealed(obj);
Object.isFrozen(obj);
```

<!-- Деструктуризация объекта -------------------------------------------------------------------------------------------------------------->

# Деструктуризация объекта

```js
// Справа существующий объект, левая сторона – шаблон
let { var1, var2 } = { var1: "var1", var2: "var2" };

let options = { title: "Menu", width: 100, height: 200 };
let { title, width, height } = options;
alert(title);
alert(width);
alert(height);

// Порядок не имеет значения
let { height, width, title } = options; //тоже самое

// В случае присваивания другой переменной
//из примера выше
let { width: w, height: h, title } = options; //двоеточия показывают что куда идет
alert(title);
alert(w);
alert(h);

// Значения по умолчанию могут быть функциями
let { width = prompt("width"), title = prompt("title") } = options;

// Могут совмещать : и =
let { width: w = 100, height: h = 200, title } = options;

// Взять то, что нужно:
let { title } = options;
```

# Глобальное запечатывание объекта

Дескрипторы свойств работают на уровне конкретных свойств.
Методы, которые ограничивают доступ ко всему объекту:

```js
Object.preventExtensions(obj); //Запрещает добавлять новые свойства в объект.
Object.seal(obj); //Запрещает добавлять/удалять свойства. Устанавливает configurable: false для всех существующих свойств.
Object.freeze(obj); //Запрещает добавлять/удалять/изменять свойства. Устанавливает configurable: false, writable: false для всех существующих свойств.

// А также есть методы для их проверки:

Object.isExtensible(obj); //Возвращает false, если добавление свойств запрещено, иначе true.
Object.isSealed(obj); //Возвращает true, если добавление/удаление свойств запрещено и для всех существующих свойств установлено configurable: false.
Object.isFrozen(obj); //Возвращает true, если добавление/удаление/изменение свойств запрещено, и для всех текущих свойств установлено configurable: false, writable: false.
```

На практике эти методы используются редко.

# Геттеры и сеттеры

Два типа свойств – data properties и accessor properties. Второй тип делится на get и set

```js
let obj = {
  get propName() {}, //срабатывает, при чтении obj.propName
  set propName(value) {}, //срабатывает при записи obj.propName = value
};

let user = {
  name: "John",
  surname: "Smith",

  get fullName() {
    return `${this.name} ${this.surname}`;
  },

  set fullName(value) {
    [this.name, this.surName] = value.split(" ");
  },
};

alert(user.fullName); // John Smith  user.fullName = "Alice Cooper"  alert(user.surName);//Cooper
```

# Дескрипторы свойств доступа

```js
// Свйоства-аксессоры не имеют value и writable, дескриптор аксессора может иметь:  get – функция для чтения,
// set – функция, принимающая один аргумент, вызываемая при присвоения свойства
// enumerable и configurable.

let user = { name: "John", surname: "Smith" };

Object.defineProperty(user, "fullName", {
  get() {
    return `${this.name}${this.surname}`;
  },

  set(value) {
    [this.name, this.surname] = value.split(" ");
  },
});
alert(user.fullName); //John Smith  for(let key in user) alert(key);
```

# геттеры и сеттеры свойств

```js
// Геттеры и сеттеры можно использовать, как обертки для реальных свойств
let user = {
  get name() {
    return this._name;
    //само значение name хранится в свойстве _name, к таким свойствам обычно на прямую не обращаются
  },
  set name(value) {
    if (name.length < 4) {
      alert("name is too short");
      return;
    }
    this._name = value;
  },
};
user.name = "Pete";
alert(user.name); //Pete  user.name = "";//too short
```

Интересная область применения аксессоров – они могут в любой момент изменить поведение обычного свойства

```js
// В примере объект со свойствами name и age
function User(name, birthday) {
  this.name = name;
  this.age = birthday;
}
// потом решили хранить свойство не age а birthday

function User(name, birthday) {
  this.name = name;
  this.birthday = birthday;
}

let john = new User("John", new Date(1992, 6, 1));
// Проблема – как поменять везде age. Добавление сеттера age решит проблему
function User(name, birthday) {
  this.name = name;
  this.birthday = birthday;

  Object.defineProperty(this, "age", {
    get() {
      let todayYear = newDate.getFullYear();
      return todayYear - this.birthday.getFullYear();
    },
  });
}
let John = new User("John", new Date(1992, 6, 1));
alert(John.birthday);
alert(John.age);
```

# Global object

```js
// Глобальный объект предоставляет переменные и функции в любом месте программы  в браузере глобальные объекты объявляются с помощью var
var gVar = 5;
alert(window.gVar); //5

// можно записывать очень важные свойства
windows.currentUser = {
  name: "John",
};
alert(currentUser.name); // John
alert(windows.currentUser.name); //John

// TG;
const a = 1;
delete a; //false
console.log(a); //1
this.b = 4;
delete b; //true
console.log(b); //undefined
```

# Reference type

```js
let user = {
  name: "John",
  hi() {
    alert(this.name);
  },
  bye() {
    alert("bye");
  },
};
user.hi(); // работает

(user.name == "John" ? user.hi : user.bye)(); //ошибка
```

(object.method())
операция – оператор точка object.method() – возвращает свойство объекта – его метод obj.method
операция – скобки () вызывают этот метод

если мы переместим операции в отдельные строки , то значение будет потерянно

```js
let user = {
  name: "John",
  hi() {
    alert(this.name);
  },
};

let hi = user.hi; //сохраняет функцию в переменной здесь нет this
hi(); //Ошибка так как значение this == undefined
```

user.hi() – точка возвращает не саму функцию, а специальное значение ссылочного типа (base, name, strict)
base – объект, name – свойства объекта, в нашем случае будет (user, "hi", true), когда () применяются к значению ссылочного типа, то ставится правильный this, при любых других операциях ссылочный тип заменяется на значение user.hi, то есть правильный this будет в случае

Сам по себе вызов (obj.method)() – кардинально ни чем не отличаются

```js
function makeUser() {
  //здесь this == undefined
  return {
    name: "John",
    ref: this,
  };
}
let user = makeUser();
alert(user.ref.name); //Ошибка Cannot read property "name" or undefined
```

```js
let user = {
  firstName: "John",
  seyHi() {
    let arrow = () => alert(this.firstName);
    arrow();
  },
};
user.seyHi(); //John
```

Правила определения this никак не смотрят на объявление объекта this внутри makeUser() == undefined, литерал не важен, таким образом ref: this берет this из makeUser()

```js
function makeUser() {
  return {
    name: "John",
    ref() {
      return this;
    },
  };
}
let user = makeUser();
alert(user.ref().name); //John – теперь user.ref() вызывается как метод и значение this будет объект перед точкой
```

```js
// TG;
"use strict";
const obj = {
  foo() {
    //console.log(this.name) Marco Polo
    function getName() {
      console.log(this.name);
    }
    getName();
  },
  name: "Marco Polo",
  callbackFoo(callback) {
    callback();
  },
};
obj.foo(); //ошибка в use strict режиме, а без use strict - пустая строка
obj.callbackFoo(function () {
  console.log(this); //undefined в use strict, без use strict объект window
});
// TG;
window.a = 777;
function f() {
  console.log(this.a);
}
const g = () => console.log(this.a);
const obj = { a: 666, f, g };
obj.f(); //666
obj.g(); //777 если window.a = 777 или var a = 777, если поменять на let то undefined
```

# spread

Копирует все enumerable свойства

```js
let options = {
  title: "Menu",
  height: 100,
  width: 200,
};

let { title, ...rest } = options;
alert(rest.height); //100
alert(rest.width); //200

// Подвох с let. JS обрабатывает { } в основном потоке кода как юлок кода
let title, width, height;

const { title, weight, height } = { title: "Menu", width: 200, height: 100 };
//исправленный вариант
({ title, width, height } = { title: "Menu", width: 200, height: 100 });

alert(title);
```

## Вложенная деструктуризация

```js
let options = {
  size: {
    width: 100,
    height: 200,
  },
  items: ["Cake", "Donut"],
  extra: true,
};

let {
  size: { width, height },
  items: [item1, item2],
  title = "Menu",
} = options;

// size и items отсутствуют так как мы взяли их содержимое
```

# this

```js
// эти объекты делают одно и то же (одинаковые методы)
user = {
  sayHi: function () {
    alert("Привет");
  },
};
// сокращённая запись
user = {
  sayHi() {
    // то же самое, что и "sayHi:  function()"
    alert("Привет");
  },
};

// this- это объект перед точкой, которой  использовался для вызова метода

let user = {
  name: "John",
  age: "30",
  sayHi() {
    alert(this.name);
  },
};
user.sayHi(); //John

// this не является фиксированным, его значение  зависит от контекста
let user = { name: "John" };
let admin = { name: "Admin" };
function sayHi() {
  alert(this.name);
}
user.f = sayHi;
admin.f = sayHi;
user.f(); //John
admin.f(); //Admin
// вызов без объекта this == undefined
function sayHi() {
  alert(this);
}
sayHi(); //undefined в данном случае ссылка на глобальный объект window
```

## Потеря this

```js
let user = {
  firstName: "Vasya",
  setHi() {
    alert(`Hi, ${this.name}!`);
  },
};
setTimeout(user.sayHi, 1000); // hi, undefined, так как setTimeout устанавливает this = window

//Решение 1 функция – обертка объект тот же
setTimeout(function () {
  user.sayHi();
}, 1000);
```

есть уязвимость – если до момента вызова setTimeout user измениться, то setTimeout вызовет его с изменениям

# toJson()

как и toString для преобразования строк, объект может содержать toJSON

```js
// Объекты Date имеют таковой
let room = {
  number: 23,
};

let meetup = {
  title: "Conference",
  date: new Date(Date.UTC(2017, 0, 1)),
  room,
};

alert(JSON.stringify(meetup));

// Добавим собственную реализацию метода toJSON в объект room

let room = {
  number: 23,
  toJSON() {
    return this.number;
  },
};

alert(JSON.stringify(room)); //23
alert(JSON.stringify(meetup)); //{"title": "Conference", "room": 23} работает, когда обращаются на прямую и когда объект room вложен
```

# Object.entries

```js
let obj = { name: "John", age: 30 };
let map = new Map(Object.entries(obj)); //Здесь Object.entries возвращает массив пар ключ-значение: [
// ["name","John"], ["age", 30] ]. Это именно то, что нужно для создания Map.

// Object.fromEntries (Map в obj) можно преобразовать коллекцию в объект
let prices = Object.formEntries([
  ["banana", 1],
  ["orange", 2],
]);
// prices = {banana:1, orange 2}  Alert(prices.orange)//2

// Чтобы получить объект из Map:
let map = new Map();
Map.set("banana", 1);
let obj = Object.fromEntries(map.entries()); //map.entries возвращает массив пар ключ/значение
Alert(object.banana); //1  Короче:
let object = Object.fromEntries(map); //убрали .entries
```

```js
let user = { name: "John", age: 30 };
Object.keys(obj); // ["name", "age"]  Object.values(obj)// ["John", 30]
Object.entries(obj); //[["name", "John"], ["age", 30]]

for (let value of Object.values(user)) {
  alert(value);
}
```

# Object.fromEntries

у объектов нет методов map, filter, но это можно сделать с помощью Object.entries с последующим вызовом Object.fromEntries

- Вызов Object.entries(obj) вызывает массив пар ключ/значение для obj
- На нем вызываем методы массива
- Используем Object.fromEntries(array) на результате, чтобы обратно вернуть его в объект

```js
let prices = { banana: 1, orange: 2, meat: 4 };

let doublePrices = Object.fromEntries(Object.entries(prices).map(([key, value] => [key, value * 2 ])))

alert( doublePrices.meat ); //8

```

# BP переиспользование переменных

- следить за тем, нет ли переменных, которые имеют одно и тоже значения, но называются по-разному

Пример 1:

```js
// если есть далее использовать id, а не обращение к someEntity
const id = someEntity.id;
```

Пример 2:

```ts
// ошибка - request.body используется во многих местах
function endpoint(request, response) {
  const id = request.body.id;
  const text = request.body.text;

  someFunction(request.body);
}
```

```ts
// исправление -  request.body используется во многих местах
function endpoint(request, response) {
  const body = request.body;

  const id = body.id;
  const text = body.text;

  someFunction(body);
}
```

<!-- Глоссарий Свойства объекта ------------------------------------------------------------------------------------------------------------>

# Глоссарий

```js
Object.create(obj,  __someProp__: {
    value: true
    // и другие дескрипторы
  }); //создает объект с прототипом obj и дескрипторами
Object.fromEntries(Object.entries(prices).map(([key, value] => [key, value ]))) //создаст объект из массива [[key, value],[key, value]]
Object.getPrototypeOf(obj); //возвращает [[Prototype]] obj
Object.getOwnPropertyNames(obj) //возвращает массив всех собственных строковых ключей.
Object.getOwnPropertySymbols(obj) // массив символьных
// получение ключе - свойств
Object.keys(obj)
Object.values(obj)
Object.entries(obj) //строковых ключей/значений/пар ключ-значение

Object.setPrototypeOf(obj, proto); //устанавливает в [[Prototype]] obj объект proto
```

```js
// методы Reflect
Reflect.ownKeys(obj); //возвращает массив всех собственных ключей.
```

методы экземпляра объекта

```js
obj.hasOwnProperty(key): //возвращает true, если у obj есть собственное (не унаследованное) свойство с именем key.
obj.hasOwn(key); // тоже самое
```
