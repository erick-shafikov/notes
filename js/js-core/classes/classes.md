# Синтаксис

```js
class MyClass {
  constructor() {
    //метод constructor() вызывается автоматически при вызове new MyClass()
    this.name = name; //создает код функции
  }
  method() {
    // метод класса
    alert(this.name); //2. Сохраняет все методы в User.prototype
  }

  field = "field"; // поле класса

  static staticField = "some-static-field"; //статичное поле

  //статичный метод
  static staticMethod() {}

  // статичный блок кода
  static {
    // будет срабатывать при инициализации класса, имеют доступ к статичным и приватным полям класса
  }

  #privateField = "some-private-field"; // приватное свойство
}
```

```js
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

- Разница в том, что класс упаковывает все методы в конструктор, при объявлении.
- Запятые между методами не ставятся
- Методы класса не перечисляемые
- функция созданная с помощью class помечена свойством [[FunctionKind]]: "classConstructor".
- Методы класса являются неперечислимыми, enumerable: false для всех методов
- Классы всегда используют use strict

<!-- конструктор ----------------------------------------------------------------------------------------------------------------------------->

# конструктор

не рекомендуется возвращать что-то из конструктора

new.target.name - позволит узнать кто создал класс

```js
class A {
  constructor() {
    console.log(new.target.name);
  }
}

class B extends A {
  constructor() {
    super();
  }
}

var a = new A(); // вернёт "A"
var b = new B(); // вернёт "B"
```

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
    set(name) {
      this._name = name;
    },
  },
});
```

# Вычисляемое свойство

```js
class User {
  ["say" + "Hi"]() {
    alert("hi");
  }
}
new User.sayHi();
```

```js
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
console.log(result); //James
// так как user.getName() возвращает this.name то при попытке получить свойства вызывается геттер get name()
```
