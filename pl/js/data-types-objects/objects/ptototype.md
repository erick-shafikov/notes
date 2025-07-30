# Прототипное наследование

[[Prototype]] – скрытое свойство которое либо null, либо ссылается на объект. Когда мы хотим прочитать свойство из object а оно отсутствует, JS берет его из прототипа это называется прототипы наследованием.

- у всех объектов прототип - Object.prototype
- у Object.prototype прототип null

```js
let animal = {
  eats: true,
  walk() {
    alert("animal Walk");
  },
};

let rabbit = { jumps: true };

rabbit.__proto__ = animal; //одним из способов задания является свойство __proto__ - геттер/сеттер для[[Prototype]];
rabbit.walk(); //"animal Walk"
```

- !!!Ссылки не могут идти по кругу
- !!!Значение proto может быть только объект или null
- !!!Прототип используется только для чтения свойств. Операции записи и удаления работают напрямую с объектом

```js
let animal = {
  eats: true,
  walk() {},
};

let rabbit = {
  __proto___: animal,
};

rabbit.walk = function () {
  alert("Rabbit! Bounce-Bounce!");
};

rabbit.walk(); //rabbit! bounce-bounce
```

# свойства асессоры

Свойства асессоры – исключения, так как запись в него обрабатывается функцией – сеттером, то есть это практически вызов функции

```js
let user = {
  name: "John",
  surname: "Smith",
  set fullName(value) {
    [this.name, this.surname] = value.split(" ");
  },
  get fullName() {
    return `${this.name} ${this.surname}`;
  },
};

let admin = {
  proto: user,
  isAdmin: true,
};

alert(admin.fullName); //John Smith
admin.fullName = "Alice Cooper";
alert(admin.name); //Alice
alert(admin.surname); //Cooper
```

<!-- значение this --------------------------------------------------------------------------------------------------------------------------->

# значение this

this в объектах – наследователях, является объектом наследователем, а не прототипом. Неважно, где находится метод: в объекте или его прототипе. При вызове метода this – всегда объект перед точкой

```js
let animal = {
  walk() {
    if (!this.isSleeping) {
      alert("I walk");
    }
  },
  sleep() {
    this.isSleeping = true;
  },
};
let rabbit = {
  name: "White Rabbit",
};
rabbit.__proto__ = animal;
rabbit.sleep(); // изменяет rabbit.isSleeping
alert(rabbit.isSleeping); //true
alert(animal.isSleeping); //undefined
```

# цикл for...in

```js
//проходит не только по собственным, но и по унаследованным
let animalRabbit = {
  eats: true,
};
let rabbit = { jumps: true, proto: animal };

alert(Object.keys(rabbit)); //jumps object.keys возвращает только собственные ключи

for (let prop in rabbit) {
  alert(prop);
} //jumps, eats

// Если унаследованные свойства нам не нежны мы можем отфильтровать из с помощью метода  obj.hasOwnProperty(key) он ← true если у obj есть собственное, не унаследованное свойство с именем key //начало – предыдущий код
for (let prop in rabbit) {
  let isOwn = rabbit.hasOwnProperty(prop);

  if (isOwn) {
    alert(`our:${prop}`);
  } else {
    alert(`Inherited ${prop}`);
  }
}
```

<!-- F.Prototype --------------------------------------------------------------------------------------------------------------------------->

# F.prototype

- F.prototype используется только при вызове new F(), после присваивания они не имеют никакого отношения друг к другу
- У каждой функции уже есть свойство "prototype", объект с единственным свойством constructor, которое ссылается на функцию – конструктор

```js
function Rabbit() {}
// Rabbit.prototype === { constructor: Rabbit };
console.log(Rabbit.prototype.constructor === Rabbit); // true

function Rabbit(name) {
  this.name = name;
  alert(name);
}

let rabbit = new Rabbit("white Rabbit");
let rabbit2 = new rabbit.constructor("Black Rabbit");
//в свойстве Rabbit.prototype есть свойство  constructor === Rabbit,
// а Rabbit в свою очередь функция-конструктор, при передачи в rabbit создается свойство конструктор и у нового объекта
```

Если в F.prototype содержится объект, оператор new устанавливает его в качестве [[prototype]] для нового объекта,
при вызове с new. F.prototype обозначает обычное свойство с именем "prototype", это еще не прототип, а обычное свойство F

```js
let animal = {
  eats: true,
};

function Rabbit(name) {
  this.name = name;
}

Rabbit.prototype = animal;
let rabbit = new Rabbit("White Rabbit");
// rabbit.__proto__=== animal, при создании объекта через New Rabbit() запиши ему animal в [[Prototype]]
alert(rabbit.eats); //true
alert(rabbit.name); //White Rabbit
```

Мы можем использовать свойство constructor существующего объекта для создания нового

```js
function Rabbit(name) {
  this.name = name;
  alert(name);
}
let rabbit = new Rabbit("White Rabbit");
let rabbit2 = new Rabbit.constructor("Black Rabbit");

// JS не гарантирует правильное значение свойства constructor. Если мы заменим прототип по умолчанию на другой объект, то свойство constructor в нем не будет

function Rabbit() {}
// изменили prototype
Rabbit.prototype = {
  jumps: true,
};

let rabbit = new Rabbit();
alert(rabbit.constructor === Rabbit); //false

// так переопределим конструктор
Rabbit.prototype = {
  jumps: true,
  constructor: Rabbit,
};
// что бы сохранить ссылку на конструктор нужно присваивать новые свойства конструктору
Rabbit.prototype.jumps = true;
```

# prototype в стрелочных функция

обычные функции имеют прототип в виде объекта, со свойством constructor, у стрелочных нет конструктора

```js
function commonFunc() {
  return "common function";
}
const arrowFunc = () => "arrow function";
console.log(commonFunc.prototype); //{constructor: ƒ} ↳ constructor: ƒ commonFunc() [[Prototype]]: Object
console.log(arrowFunc.prototype); //undefined
```

# наследование

```js
// родительский класс
function Person(first, last, age, gender, interests) {
  this.name = {
    first,
    last,
  };
  this.age = age;
  this.gender = gender;
  this.interests = interests;
}

// метод
Person.prototype.greeting = function () {
  alert("Hi! I'm " + this.name.first + ".");
};

//наследование
function Teacher(first, last, age, gender, interests, subject) {
  // вызов Person - родительского класса
  Person.call(this, first, last, age, gender, interests);

  this.subject = subject;
}

//устанавливаем для наследования методов
Teacher.prototype = Object.create(Person.prototype);

Object.defineProperty(Teacher.prototype, "constructor", {
  value: Teacher,
  enumerable: false, // false, чтобы данное свойство не появлялось в цикле for in
  writable: true,
});
```

# Встроенные прототипы

```js
let obj = {};
// на самом деле obj = new Object() где Object - функция конструктор где

Object.prototype = {
  constructor: Object,
  toString: function
}
// obj.toString === obj.__proto__.toString === Object.prototype.toString
Object.prototype.__proto__ === null
```

иерархия

```js
// на верху null
console.log(Object.prototype.__proto__); //null;
console.log(Array.prototype); //Object;
console.log(Function.prototype); //Object;
console.log(Number.prototype); //Object;
```

Можно изменять внутренние прототипы

```js
String.prototype.show = function () {
  alert(this);
};

"BOOM!".show(); // BOOM!

if (!String.prototype.repeat) {
  String.prototype.repeat = function (n) {
    return new Array(n + 1).join(this);
  };
}

alert("La".repeat(3)); // LaLaLa
```

## Методы прототипов

Поверхностное клонирование

```js
let clone = Object.create(
  Object.getPrototypeOf(obj),
  Object.getOwnPropertyDescriptors(obj)
);

// словарь
let obj = Object.create(null);
```

свойство proto считается устаревшим

```js
Object.create(proto, descriptors); //создает пустой объект со свойством [[Prototype]] казанным как proto и необязательными дескрипторами свойств descriptors в него можем добавить дополнительные свойства
Object.getPrototypeOf(obj); // возвращает свойство [[Prototype]] объекта obj
Object.setPrototypeOf(obj, proto); //устанавливает свойство [[Prototype]] объекта obj как proto
```

```js
let animal = { eats: true };
let rabbit = Object.create(animal, {
  jumps: {
    value: true,
  },
});

let animal = {
  eats: true,
};

let rabbit = Object.create(animal);
alert(rabbit.eats); //true

alert(Object.getPrototypeOf(rabbit) === animal);

Object.setPrototypeOf(rabbit, {}); //заменяем prototype Объекта rabbit на {}

// Object.create можно использовать для продвинутого копирования объектов
let clone = Object.create(
  Object.getPrototypeF(obj),
  Object.getOwnPropertyDescriptors(obj)
);
```
