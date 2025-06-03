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

<!-- приватные поля класса --------------------------------------------------------------------------------------------------------------------->

# приватные поля класса

- Приватные свойства и методы должны начинаться с «#»
- Они доступны только внутри класса
- обращение к переменным с \# является ошибкой

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

# статичный блок

- блок ода выполняем при инициализации

```js
class ClassWithStaticInitializationBlock {
  static staticProperty1 = "Property 1";
  static staticProperty2;
  static {
    this.staticProperty2 = "Property 2";
  }
}

console.log(ClassWithStaticInitializationBlock.staticProperty1);
// Expected output: "Property 1"
console.log(ClassWithStaticInitializationBlock.staticProperty2);
// Expected output: "Property 2"
```

```js
class MyClass {
  static field1 = console.log("static field1");
  static {
    console.log("static block1");
  }
  static field2 = console.log("static field2");
  static {
    console.log("static block2");
  }
}
// 'static field1'
// 'static block1'
// 'static field2'
// 'static block2'
```
