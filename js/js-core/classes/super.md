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

super в обычных объектах

```js
var obj1 = {
  method1() {
    console.log("method 1");
  },
};

var obj2 = {
  method2() {
    super.method1();
  },
};

Object.setPrototypeOf(obj2, obj1);
obj2.method2(); // выведет "method 1"
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
