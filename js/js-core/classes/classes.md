# Синтаксис

```js
class MyClass {
  constructor() {
    //метод constructor() вызывается автоматически при вызове new MyClass()
    //создает код функции
    this.name = name;
  }
  method() {
    // метод класса
    //2. Сохраняет все методы в User.prototype
    alert(this.name);
  }
  field = "field"; // поле класса

  static staticField = "some-static-field"; //статичное поле

  static staticMethod() {
    //статичный метод
  }

  static {
    // статичный блок кода
    // будет срабатывать при инициализации класса, имеют доступ к статичным и приватным полям класса
  }

  #privateField = "some-private-field"; // приватное свойство
}

// вызов без new приведет к ошибке
// Создается новый объект, он будет взят из прототипа
// constructor запускается с заданными аргументами и сохраняет его в this.name
const instance = new MyClass();

alert(MyClass === MyClass.prototype.constructor); //true
alert(MyClass.prototype.method); //alert( this.name );
alert(Object.getOwnPropertyNames(MyClass.prototype)); // constructor, sayHi

(typeOf MyClass) //function в JS класс – разновидность функции
```

это аналогично

```js
function MyClass() {
  this.field = "field";
}

MyClass.staticField = "some-static-field";
MyClass.staticMethod = function () {
  //статичный метод
};
MyClass.prototype.method = function () {
  // метод класса
};

(function () {
  // статичный блок кода
})();
```

```js
const MyClass = class {
  // так тоже можно создавать класс
};
```

Разница в том, что класс упаковывает все методы в конструктор, при объявлении.

!!!Запятые между методами не ставятся
!!! Методы класса не перечисляемые

различие в том, что:

- функция созданная с помощью class помечена свойством [[FunctionKind]]: "classConstructor". В отличае от обычных функций, класс не может быть вызван без new
- Методы класса являются неперечислимыми, enumerable: false для всех методов
- Классы всегда используют use strict

<!-- конструктор ----------------------------------------------------------------------------------------------------------------------------->

## конструктор

не рекомендуется возвращать что-то из конструктора

<!-- Class Expression ------------------------------------------------------------------------------------------------------------------------>

# Class Expression

```js
let User = class {
  sayHI() {
    alert("Hi");
  }
};

// NFE для классов (NCE)
let User = class MyClass {
  sayHI() {
    alert(MyClass); //код функции
  }
};

new User().sayHi();

alert(MyClass); //MyClass переменная видна только внутри кода функции
// Динамическое создание классов
function makeClass(phrase) {
  return class {
    sayHi() {
      alert(phrase);
    }
  };
}
let User = makeClass("Привет");
new User().SayHi();
```

<!-- геттеры и сеттеры ----------------------------------------------------------------------------------------------------------------------->

# геттеры и сеттеры

```js
class User {
  constructor(name) {
    this._name = name;
  }
  get name() {
    return this._name;
  }
  set name(value) {
    if (value.length < 4) {
      alert("too short");
      return;
    }
    this._name = value;
  }
}
let user = new User("Ivan");
alert(user.sayHi());
let user = new User(""); //2 short
```

При объявлении класса геттеры/сеттеры создаются в User.prototype

```js
Object.defineProperties(User.prototype, {
  name: {
    get() {
      return this._name;
    },
    set(name) {},
  },
});
```

## Вычисляемое свойство

```js
class User {
  ["say" + "Hi"]() {
    alert("hi");
  }
}
new User.sayHi();

// TG
class User {
  constructor(name) {
    this.name = name;
  }
  get name() {
    return "James";
  }
  set name(value) {}
  getName() {
    return this.name;
  }
}
const user = new User("Brendan");
const result = user.getName();
console.log(result); //James так как user.getName() возвращает this.name то при попытке получить свойства вызывается геттер get name()
```

<!-- Inheritance ----------------------------------------------------------------------------------------------------------------------------->

# Inheritance

```js
class Animal {
  //есть класс Animal
  constructor(name) {
    this.speed = 0;
    this.name = name;
  }
  run(speed) {
    this.speed = speed;
    alert(`${this.name} бежит со скоростью ${this.speed}`);
  }
  stop() {
    this.speed = 0;
    alert(`${this.name} стоит`);
  }
}

let animal = new Animal("Мой питомец");

class Rabbit {
  //есть класс Rabbit
  constructor(name) {
    this.name = name;
  }
  hide() {
    alert(`${this.name} прячется`);
  }
}

let rabbit = new Rabbit("Мой кролик");
//А теперь Rabbit расширит Animal

class Animal {
  constructor(name) {
    this.speed = 0;
    this.name = name;
  }
  run(speed) {
    this.speed = speed;
    alert(`${this.name} бежит со скоростью ${this.speed}}`);
  }
  stop() {
    this.speed = 0;
    alert(`${this.name} стоит`);
  }
}
let animal = new Animal("мой питомец");

class Rabbit extends Animal {
  //устанавливает Rabbit.prototype.[[Prototype]]
  // в Animal.prototype. после extends могут быть любые выражения
  constructor(name) {
    this.name = name; //можно убрать
  }
  hide() {
    alert(`${this.name} прячется!`);
  }
}
let rabbit = new Rabbit("White Rabbit");

// После extends разрешены любые выражения
```

<!-- super ----------------------------------------------------------------------------------------------------------------------------------->

# super

- super.method() вызывает родительский метод
- super() вызывает родительский конструктор
- Если мы определим свой метод в наследующем классе, то он заменит родительский
- нельзя получить this до вызова super

  ```js
  class Animal {
    //кролик прячется при остановке
    constructor(name) {
      this.speed = 0;
      this.name = name;
    }
    run(speed) {
      this.speed = speed;
      alert(`${this.name} run with speed ${this.speed}`);
    }
    stop() {
      this.speed = 0;
      alert(`${this.name} is stay`);
    }
  }
  ```

```js
class Rabbit extends Animal {
  hide() {
    alert(`${this.name} is hide!`);
  }
  stop() {
    super.stop(); //вызывает родительский метод this.hide();
  }
}

let rabbit = new Rabbit("Белый кролик");
rabbit.run(5); // Белый кролик бежит со скоростью 5
rabbit.stop(); // Белый кролик стоит. Белый кролик прячется
```

!!!У стрелочных функций нет super

```js
class Rabbit extends Animal {
  stop() {
    setTimeout(() => super.stop(), 1000); // вызывает родительский метод через 1 секунду? ,берется из  родительской
  }
}

setTimeout(function () {
  super.stop();
}, 1000); //ошибка
```

# Переопределение конструктора

Если класс расширяет другой класс, в котором нет конструктора, то создается конструктор вида:

```js
class Animal {
  constructor(name) {
    this.speed = 0;
    this.name = name;
  }
}

class Rabbit extends Animal {
  constructor(...args) {
    super(...args);
  }
}

class Rabbit extends Animal {
  constructor(name, earLength) {
    this.speed = 0;
    this.name = name;
    this.earLength = earLength;
  }
}
let rabbit = new Rabbit("white", 10); //error this is not defined

class Animal {
  constructor(name) {
    this.speed = 0;
    this.name = name;
  }
}

class Rabbit extends Animal {
  constructor(name, earLength) {
    super(name);
    //Наследующий класс функция – конструктор помечена специальным внутренним свойством  [ConstructionKind]]:derived
    //Когда выполняется обычный конструктор он создает пустой объект и присваивает его this
    //Когда запускается конструктор унаследованного класса он этого не делает, он ждет, что это сделает  конструктор родительского класса

    this.earLength = earLength;
  }
}

let rabbit = new Rabbit("White Rabbit", 10);
alert(rabbit.name); //White Rabbit
alert(rabbit.earLength); //10
```

В классах потомках конструктор обязан вызывать super до использования this

# Свойство [[HomeObject]]

```js
// Пример, где вроде бы работает
let animal = {
  name: "Animal",
  eat() {
    alert(`${this.name} ест`);
  },
};

let rabbit = {
  proto: animal,
  name: "Кролик",
  eat() {
    this.__proto__.eat.call(this); // для правильного определения контекста
  },
};

rabbit.eat(); //Кролик ест
```

```js
// Пример, где вроде не работает
let animal = {
  name: "Animal",
  eat() {
    alert(`${this.name} ест`);
  },
};

let rabbit = {
  proto: animal,
  name: "Кролик",
  eat() {
    this.__proto__.eat.call(this);
    //при вызове здесь, метод вызывает себя же
    // для правильного определения контекста
  },
};

let longEar = {
  proto: rabbit,
  eat() {
    this.__proto__.eat.call(this);
    //this == longEar, тогда this.proto == rabbit
  },
};

longEar.eat(); //Error Max call stack
```

Когда функция объявлена как метод внутри класса или объекта ее свойство [[HomeObject]] становится равно этому объекту. Затем super использует его, чтобы получить прототип родителя и его методы

```js
let animal = {
  name: "Animal",
  eat() {
    //[[HomeObject]]==animal
    alert(`${this.name} is eating`);
  },
};
let rabbit = {
  proto: animal,
  name: "Rabbit",
  eat() {
    //rabbit.eat.[[HomeObject]] == rabbit
    super.eat(); //[HomeObject]] == rabbit
  },
};
let longEar = {
  proto: rabbit,
  name: "LongEar",
  eat() {
    super.eat(); //[HomeObject]] == longEar
  },
};
longEar.eat(); //longEar is eating
// Метод запоминают свои объекты с помощью свойства [[HomeObject]]

// Единственно место, где используется [[HomeObject]] – это super, без super – метод свободный с super уже нет
// Пример неверного результата super
let animal = {
  sayHi() {
    console.log("Я животное");
  },
};
let rabbit = {
  __proto__: animal,
  sayHi() {
    super.sayHi(); //[[HomeObject]] == rabbit, он создан в rabbit
  },
};
let plant = {
  sayHi() {
    console.log("Я растение");
  },
};
let tree = {
  __proto__: plant,
  sayHi: rabbit.sayHi,
};
tree.sayHi(); // я животное, так как в нем есть super.sayHi()
```

## Методы не свободны

```js
// Метод, а не свойства – функции
// в функциях – методах нет [[HomeObject]]

let animal = {
  eat: function () {
    // не eat(){ ... }
  },
};

let rabbit = {
  __proto__: animal,
  eat: function () {
    super.eat();
  },
};

rabbit.eat(); //Ошибка вызова super "super" keyword unexpected here
```

<!-- статические методы и свойства ----------------------------------------------------------------------------------------------------------->

# статические методы и свойства

Мы можем присвоить методы самой функции - классу, а не её prototype – метод, который стоит над всеми
наследующими объектами

```js
class User {
  static staticMethod() {
    alert(this === User);
  }
}
User.staticMethod(); //true
// Это тоже самое (фактически), что присвоить метод напрямую
class User {}
User.staticMethod = function () {};

//this при вызове User.staticMethod() является сам конструктор класса.
// Обычно используются только для  классов. Есть объекты статей Article и нужна функция их сравнения

class Article {
  constructor(title, data) {
    this.title = title;
    this.date = date;
  }
  static compare(articleA, articleB) {
    return articleA.date - articleB.date;
  }
}

let articles = [
  new Article("HTML", new Date(2019, 1, 1)),
  new Article("CSS", new Date(2019, 0, 1)),
  new Article("JS", new Date(2019, 11, 1)),
];

articles.sort(Article.compare);
```

Пример фабричного метода. Нужно создавать статьи
Создание через заданные параметры//конструктор
Создание пустой статьи с сегодняшней датой:

```js
class Article {
  constructor(title, date) {
    this.title = title;
    this.date = date;
  }
  static createTodays() {
    //this == Article
    return new this("today's article", new Date());
  }
}

let article = Article.createTodays();
alert(article.title); //сегодняшний дайджест

// Статистические свойства
class Article {
  static publisher = "Name";
}
```

## Наследование статического свойства

```js
class Animal {
  //метод Animal.compare наследуется и доступен как Rabbit.compare
  constructor(name, speed) {
    this.speed = speed;
    this.name = name;
  }

  run(speed = 0) {
    this.speed += speed;
    alert(`${this.name} run with speed ${this.speed}`);
  }
  static compare(animalA, animalB) {
    return animalA.speed - animalB.speed;
  }
}

class Rabbit extends Animal {
  hide() {
    alert(`${this.name} is hiding`);
  }
}

let rabbits = [new Rabbit("White", 10), new Rabbit("Black", 5)];

rabbit.sort(Rabbit.compare);

rabbits[0].run(); //black

alert(Rabbit.proto === Animal); //true
alert(Rabbit.prototype.proto === Animal.prototype); //true
```

Rabbit extends Animal создает две ссылки на прототип:
Функция Rabbit прототипно наследуется от Animal
Rabbit.prototype прототипно наследует от Animal.prototype class Animal {}
class Rabbit extends Animal{}

```js
// Все объекты наследуют от Object.prototype и имеют доступ к общим методам
class Rabbit {
  constructor(name) {
    this.name = name;
  }
}
let rabbit = new Rabbit("Rab");
alert(rabbit.hasOwnProperty("name")); // true

//С ошибкой:
class Rabbit extends Object {
  constructor(name) {
    this.name = name;
  }
}

let rabbit = new Rabbit("Rab2");
alert(rabbit.hasOwnProperty("name")); //Ошибка
```

```js
class Rabbit extends Object {
  constructor(name) {
    super();
    this.name = name;
  }
}

let rabbit = new Rabbit("Rab2");
alert(rabbit.hasOwnProperty("name")); // true
```

После исправления есть различие между class Rabbit и class Rabbit extends Object

```js
class Rabbit extends Object {}
alert(Rabbit.prototype.__proto__ === Object.prototype); //true наследование между prototype функции-конструкторов
alert(Rabbit.__proto__ === Object); //true наследование между функциями конструкторов
alert(Rabbit.getOwnPropertyNames({ a: 11, b: 2 })); //Rabbit представляет доступ к статистическим методам Object через Rabbit
```

Но если не унаследовать явно то для Rabbit. proto не установлен Object

```js
class Rabbit {}

alert(Rabbit.prototype.__proto === Object.prototype); //true  alert( Rabbit. proto	=== Object); //false
alert(Rabbit.proto === Function.prototype); //как у каждой функции по умолчанию

alert(Rabbit.getOwnPropertyNames({ a: 1, b: 2 })); // Error

// Все из-за того, что
Object.__proto__ === Function.prototype;
```

class Rabbit: Rabbit.**proto** === Function.prototype
class Rabbit extends Object Rabbit.**proto** === Object //нужно указать super в конструкторе

<!-- Приватные поля ------------------------------------------------------------------------------------------------------------->

# приватные поля

Поле – это свойство или метод объекта: публичные и приватные (доступные только внутри класса)

```js
class CoffeeMachine {
  waterAmount = 0;

  constructor(power) {
    this.power = power;
    alert(`this CM with power of ${power}`);
  }
}

let coffeeMachine = new CoffeeMachine(100);
coffeeMachine.waterAmount = 200;

class CoffeeMAchine {
  _waterAmount = 0; //Защитим свойства с помощью префикса._
  set waterAmount(value) {
    if (value < 0) throw new Error("Negative volume of water");
    this._waterAmount = value;
  }
  get waterAmount() {
    return this._waterAmount;
  }
  constructor(power) {
    this._power = power;
  }
}
coffeeMachine.waterAmount = -10; //Error
// Свойство только для чтения
class CoffeeMAchine {
  // но лучше выбирать функции getPower() и setPower() т.к поддерживают несколько арг
  constructor(power) {
    this.power = power;
  }
  get power() {
    return this._power;
  }
  //теперь power нельзя изменить, так как нет сеттера
}
```

<!-- защищенное свойство --------------------------------------------------------------------------------------------------------------------->

# Защищенное свойство

Приватные свойства и методы должны начинаться с «#» Они доступны только внутри класса

```js
class CoffeeMachine {
  #waterLimit = 200;
  #checkWater(value) {
    if (value < 0) throw new Error("Negative water volume");
    if (value > this.#waterLimit) throw new Error("to much");
  }
}

let coffeeMachine = new CoffeeMAchine();

coffeeMachine.#checkWater(); //error  coffeeMachine.WaterLimit = 100//error
```

Оба свойства могут быть в объекте

```js
class CoffeeMAchine {
  #waterAmount = 0;
  get waterAmount() {
    return this.#waterAmount;
  }
  set waterAmount(value) {
    if (value < 0) throw new Error("Negative");
    this.#waterAmount = value;
  }
}
// при наследовании мы не получим прямого доступа
class MegaCoffeeMachine extends CoffeeMachine {
  method() {
    alert(this.#waterAmount); //Error: can only access from CoffeeMAchine
  }
}

// Можно получить доступ к свойству через this[name], но с приватным полем this["#name"] не работает

class User {
  sayHi() {
    let fieldName = "name";
    alert(`Hello, ${this[fieldName]}`);
  }
}
```

дублирование или имя без # в конструкторе приведет к ошибке

# Расширение встроенных классов

От встроенных Map, Array тоже можно наследовать

```js
class PowerArray extends Array {
  // arr.constructor === PowerArray
  isEmpty() {
    return this.length === 0;
  }

  static get [Symbol.spaces]() {
    return Array;
  } //с помощью этого метода такие методы ка map, filter будут возвращать не powerArray а обычные Array без расширенных методов
}

let arr = new PowerArray(1, 2, 5, 10, 50);
alert(arr.isEmpty()); //false

let filteredArr = arr.filter((item) => item >= 10);
alert(filteredArr); //10,50
alert(filtered.isEmpty()); //false
// Поэтому при вызове arr.filter() он внутри создает массив результатов именно используя arr.constructor а не
// обычный массив? чтобы возвращал обычные массивы такие методы как map, filter
class PowerArray extends Array {
  isEmpty() {
    return this.length === 0;
  }

  static get [Symbol.spaces]() {
    return Array;
  }
}
let arr = new PowerArray(1, 2, 5, 10, 50);
let filteredArr = arr.filter((item) => item >= 10);
alert(filteredArr.isEmpty()); // Error: filteredArr.isEmpty is not a function
```

У встроенных объектов есть собственные статические методы – Object.keys Array.isArray ..
Встроенные классы не наследуют статические методы друг друга

Array и Date наследуют от Object, так что в их экземплярах доступны методы из Object.prototype,
Array.[[Prototype]] не ссылается на Object так что нет методов Array.keys()

нет связи между Date и Object они независимы только Date.prototype наследуется от Object.prototype в это разница с extends

<img src='./assets/js/class-static-inheritance.png'/>

```js
let arr = [10, 20, 30, 40, 50];
Array.prototype.each = function () {};
for (let i in arr) {
  console.log(i); //0 ,1, 2, 3, 4, each for..in итерируется по всем полям объекта и его прототипов (т.е. проходит по всей цепочке прототипов).
}
```

<!-- instanceOf ------------------------------------------------------------------------------------------------------------------------------>

# instanceOf

obj instanceOf class ← true если obj принадлежит классу Class или наследующему от него

```js
class Rabbit{}
let rabbit = new Rabbit();
alert( rabbit instanceOf Rabbit) //true

// так же работает с функциями – конструкторами
function Rabbit() {}
alert( new Rabbit() instanceOf Rabbit); //true

```

Для изменения поведения instanceof используется статистический метод Symbol.hasInstance
при проверке instanceof, если есть метод Symbol.hasInstance, то вызывать его, он ← возвращает либо true
либо false. Проверяет он prototype"ы

```js
class Animal {
  static [Symbol.hasInstance](obj) {
    if (obj.canEat) return true;
  }
}

let obj = { canEat: true };
alert(obj instanceof Animal); //true
// в противном случае проводится проверка

obj.__proto__=== Class.prototype?
obj.__proto__.  proto === Class.prototype?
obj.__proto__. proto .__proto === Class.prototype?

```

<!-- ObjA.isPrototypeOf(objB) ---------------------------------------------------------------------------------------------------------------->

# ObjA.isPrototypeOf(objB)

ObjA.isPrototypeOf(objB) ← true если objA есть где-то в прототипной цепочке objB.

```js
function Rabbit() {}
let rabbit = new Rabbit();
Rabbit.prototype = {}; //обнуляем прототип конструктора  alert (rabbit instanceof Rabbit);//false
```

<!-- Object.prototype.toString --------------------------------------------------------------------------------------------------------------->

# Object.prototype.toString

Обычные объекты преобразуются к строке как [object Object]
Object.prototype.toString возвращает тип

```js
let objectToString = Object.prototype.toString;
let arr = [];
alert(objectToString.call(arr)); //[object Array] а call здесь для контекста this = arr

let s = Object.prototype.toString;

alert(s.call(123)); //[object Number]
alert(s.call(null)); //[object Null]
alert(s.call(alert)); //[Object function]

let user = {
  // Поведение метода toString можно настраивать через специальное свойство Symbol.toStringTag
  [Symbol.toStringTag]: "User",
};

alert({}.toString.call(user)); //[object User]

alert(window[Symbol.toStringTag]); //window
alert(XMLHttpRequest.prototype[Symbol.toStringTag]); //XMLHttpRequest
alert({}[Symbol.toStringTag].call(window)); //[object Window]
alert([Symbol.toStringTag].call(new XMLHttpRequest())); //[object XMLHttpRequest]
```

<!-- Примеси --------------------------------------------------------------------------------------------------------------------------------->

# Примеси

Примесь – это класс, методы которого предназначены для использования в других классах, причем без наследования от примеси

```js
let sayHiMixin = {
  //примесь
  sayHi() {
    alert("Hi ${this.name}");
  },
  sayBye() {
    alert("Bye {this.name}");
  },
};

class User {
  constructor(name) {
    this.name = name;
  }
}

Object.assign(User.prototype, sayHiMIxin); //Копируется user.prototype
new User("Vasya").sayHi(); //Hi Vasya
```

Это не наследование, а просто копирование методов.
User может наследовать от другого класса, но при этом также включать в себя примеси, подмешивающие другие методы

```js
class User extends Person {}
Object.assign(User.prototype, sayHiMixin);
```

Примеси могут наследовать друг друг

```js
let sayMixin = {
  //примесь
  say(phrase) {
    alert(phrase);
  },
};

let sayHiMixin = {
  __proto__: sayMixin,

  sayHi() {
    super.say(`Hi ${this.name}`);
  },

  sayBye: () => {
    super.say(`Bye ${this.name}`);
  },
};

class User {
  constructor(name) {
    this.name = name;
  }
}

Object.assign(User.prototype, sayHiMixin);
new User("John").sayHi(); //Hi John

// при вызове родительского метода super.say() из sayHiMixin этот метод ищется в прототипе самой примеси, а не класса
```

```js
let EventMixin = {
  on(eventName, handler) {
    //обработчик для события с заданным имя, получив данные из trigger
    if (!this._eventHandlers) this._eventHandlers = {};
    if (!this._eventHandlers[eventName]) {
      this._eventHandler[eventName] = [];
    }
    this._eventHandlers[eventName].push(handler);
  },
  off(eventName, handler) {
    //удаляет обработчик
    let handlers = this._eventHandlers && this._eventHandlers[eventName];
    if (!handlers) return; //если такого нет, то выход
    for (let i = 0; i < handlers.lenght; i++) {
      if (handlers[i] === handler) {
        handlers.splice(i--, 1);
      }
    }
  },
  trigger(eventName, ...args) {
    //для генерации события, name – имя события, далее доп аргументы [...arg]
    if (!this._eventHandlers || !this._eventHandlers[eventName]) {
      return;
    }
    this._eventHandlers[eventName].forEach((handler) =>
      handler.apply(this, args)
    );
  },
};

class Menu {
  choose(value) {
    this.trigger("select", value);
  }
}

Object.assign(Menu.prototype, eventMixin);
let menu = new Menu();

menu.on("select", (value) => alert(`Выбранное значение: ${value}`));

menu.choose("123"); //Выбранное значение 123
```
