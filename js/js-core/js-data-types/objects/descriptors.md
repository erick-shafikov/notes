У каждого свойства есть три атрибута (флага):

- writable – если true, то свойство можно изменить
- enumerable – если true, то свойство можно перечислять в циклах
- configurable – если true, свойство можно удалить, а эти атрибуты можно изменять, иначе этого делать нельзя. При создании свойств, все флаги – true.

<!-- getOwnPropertyDescriptor ---------------------------------------------------------------------------------------------------------------->

# getOwnPropertyDescriptor

Чтобы получить полную информацию о свойстве, все дескрипторы свойств сразу, можно воспользоваться методом Object.getOwnPropertyDescriptors(obj).

```js
let descriptor = Object.getOwnPropertyDescriptor(obj, propertyName);
// где obj – Объект,
// propertyName – имя свойства,
// возвращаемый объект – дескриптор свойства
let user = {
  name: "John",
};

let descriptor = Object.getOwnPropertyDescriptor(user, "name");
alert(JSON.stringify(descriptor, null, 2));
// {"value":john,  "writable":true,  "enumerable":true ,  "configurable":"true"}
```

<!-- defineProperty -------------------------------------------------------------------------------------------------------------------------->

# defineProperty

Чтобы изменить

```js
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

## копирование объекта

Вместе с Object.defineProperties этот метод можно использовать для клонирования объекта вместе с его флагами:

```js
let clone = Object.defineProperties({}, Object.getOwnPropertyDescriptors(obj));
// Обычно при клонировании объекта мы используем присваивание, чтобы скопировать его свойства:

for (let key in user) {
  clone[key] = user[key];
}

// Но это не копирует флаги. Так что если нам нужен клон «получше», предпочтительнее использовать
// Object.defineProperties.
```

- for..in игнорирует символьные свойства, а Object.getOwnPropertyDescriptors возвращает дескрипторы всех свойств, включая свойства-символы.

## неконфигурируемое свойство

```js
let descriptor = Object.getOwnPropertyDescriptor(Math, "PI");
alert(Json.stringify(descriptor, null, 2));
// Определение свойства, как не конфигурируемого – это дорога в один конец, его нельзя будет изменить
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

## Глобальное запечатывание объекта

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

<!-- дескрипторы свойств доступа ------------------------------------------------------------------------------------------------------------->

# дескрипторы свойств доступа

```js
// Свойства -асессоры не имеют value и writable, дескриптор асессора может иметь:
// get – функция для чтения,
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

Интересная область применения асессоров – они могут в любой момент изменить поведение обычного свойства

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
