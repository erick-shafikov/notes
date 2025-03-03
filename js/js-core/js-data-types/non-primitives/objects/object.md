<!-- Создание объекта ---------------------------------------------------------------------------------------------------------------------->

# Создание объекта

Объекты - это область памяти, ссылка или указатель

Варианты создания объектов

```js
// литеральный тип
const obj = {};
let user1 = new Object(); // синтаксис "конструктор объекта"
const obj = new ObjFunctionCreator(); // через функции конструкторы ObjFunctionCreator = function(){this. = ....}
// Object.create можно задать прототип объекта
Object.create(proto, [descriptors]);
// fromEntries
Object.fromEntries([
  ["key0", "value0"],
  ["key1", "value1"],
]);

Object.create(obj,  __someProp__: {
    value: true
    // и другие дескрипторы
  }); //создает объект с прототипом obj и дескрипторами
```

# Добавление свойств

Ключами могут быть и пустые строки и знак !, все что может быть преобразовано к строке

```js
let key = "some key"; // то же самое, что и obj["some key"] = true;
obj[key] = true;
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

# Удаление свойств

```js
delete obj.age;
delete obj["some key"];
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

# методы

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

Проверка наличия свойства:

```js
// Проверка наличия свойства:
let user = { age: 30 };
let key = "age";
alert(key in user); // true
```

# Перебор

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

# копирование объектов

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

<!-- геттеры и сеттеры ----------------------------------------------------------------------------------------------------------------------->

# геттеры и сеттеры

Два типа свойств – data properties и accessor properties. Второй тип делится на get и set

```js
let obj = {
  get propName() {
    //срабатывает, при чтении obj.propName
  },
  set propName(value) {
    //срабатывает при записи obj.propName = value
  },
};
```

```js
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
```

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
```

<!-- Методы Object --------------------------------------------------------------------------------------------------------------------------->

# Методы Object

## Object.is()

строгое сравнение двух объектов

возвращает false для сравнения -0 и +0, true для сравнений двух NaN

Object.is(obj1, obj2)

## Object.entries(obj)

```js
let obj = { name: "John", age: 30 };
let map = new Map(Object.entries(obj)); //Здесь Object.entries возвращает массив пар ключ-значение: [
// ["name","John"], ["age", 30] ]. Это именно то, что нужно для создания Map.
```

## Object.keys(obj)

```js
let user = { name: "John", age: 30 };
Object.keys(obj); // ["name", "age"]  Object.values(obj)// ["John", 30]
```

## Object.fromEntries

у объектов нет методов map, filter, но это можно сделать с помощью Object.entries с последующим вызовом Object.fromEntries

- Вызов Object.entries(obj) вызывает массив пар ключ/значение для obj
- На нем вызываем методы массива
- Используем Object.fromEntries(array) на результате, чтобы обратно вернуть его в объект

```js
let prices = { banana: 1, orange: 2, meat: 4 };

let doublePrices = Object.fromEntries(Object.entries(prices).map(([key, value] => [key, value * 2 ])))

// Object.fromEntries (Map в obj) можно преобразовать коллекцию в объект
let prices = Object.formEntries([
  ["banana", 1],
  ["orange", 2],
]);
// prices = {banana:1, orange 2}  Alert(prices.orange)//2

```

```js
// Чтобы получить объект из Map:
let map = new Map();
Map.set("banana", 1);
let obj = Object.fromEntries(map.entries()); //map.entries возвращает массив пар ключ/значение
let object = Object.fromEntries(map); //убрали .entries
```

## Object.getPrototypeOf

возвращает [[Prototype]] obj

```js
Object.getPrototypeOf(obj);
```

## Object.getOwnPropertyNames

возвращает массив всех собственных строковых ключей.

```js
Object.getOwnPropertyNames(obj);
```

## Object.getOwnPropertySymbols

массив символьных

```js
Object.getOwnPropertySymbols(obj);
```

## Object.setPrototypeOf

устанавливает в [[Prototype]] obj объект proto

```js
Object.setPrototypeOf(obj, proto);
```

<!-- Методы объекта -------------------------------------------------------------------------------------------------------------------------->

# методы экземпляра объекта

## hasOwnProperty и hasOwn

```js
obj.hasOwnProperty(key): //возвращает true, если у obj есть собственное (не унаследованное) свойство с именем key.
obj.hasOwn(key); // тоже самое
```

## toJson()

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

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs:

## BP переиспользование переменных

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

## Bps linked list

```js
let list = {
  value: 1,
  next: {
    value: 2,
    next: {
      value: 3,
      next: {
        value: 4,
        next: null,
      },
    },
  },
};
// Разделить связанный список
let secondList = list.next.next;
list.next.next = null;
// Обледенить:
list.next.next = secondList;

// для добавления нового:
list = { next: list };

// Вывод по порядку(цикл):
function printList(list) {
  let tmp = list;
  while (tmp) {
    alert(tmp.value);
    tmp = tmp.next;
  }
}
// Вывод по порядку(рекурсия):
function printList(list) {
  alert(list.value); // выводим
  // текущий элемент
  if (list.next) {
    printList(list.next); //делаем то же самое для остальной  части списка
  }
}
// Вывод в обратном(рекурсия):
function printReverseList(list) {
  if (list.next) {
    printReverseList(list.next);
  }
  alert(list.value);
}
if (obj.next != null) {
  revPrintList(obj.next);
  alert(obj.value);
} else {
  alert(obj.value);
}
```
