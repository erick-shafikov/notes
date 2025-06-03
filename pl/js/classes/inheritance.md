<!-- наследование ----------------------------------------------------------------------------------------------------------------------------->

# наследование

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

- наследоваться от объекта нельзя с помощью extends, только с помощью Object.setPrototypeOf();

```js
var Animal = {
  speak() {
    console.log(`${this.name} издаёт звук.`);
  },
};

class Dog {
  constructor(name) {
    this.name = name;
  }
}

// Если вы этого не сделаете, вы получите ошибку TypeError при вызове speak.
Object.setPrototypeOf(Dog.prototype, Animal);

let d = new Dog("Митци");
d.speak(); // Митци издаёт звук.
```

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
