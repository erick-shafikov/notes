# Proxy

- прокси-объект - обертка над объектом целью
- обработчики - объект с ловушками
- ловушки - методы переопределяющий поведение объекта
- цель - исходный объект

Объект proxy оборачивается вокруг объекта и может перехватывать разные действия с ним. Синтаксис:

```js
let target = {};
let proxy = new Proxy(target, {}); //пустой handler

proxy.test = 5; //записали в proxy

alert(proxy.test); //5

for (let key in proxy) {
  alert(key); //test
}
```

# apply

ловушка для вызова функций

- target- объект
- thisArg - аргумент this для вызова
- argumentsList -список аргументов

```js
function sum(a, b) {
  return a + b;
}

const handler = {
  apply: function (target, thisArg, argumentsList) {
    console.log(`Calculate sum: ${argumentsList}`);
    // "Calculate sum: 1,2"

    return target(argumentsList[0], argumentsList[1]) * 10;
  },
};

const proxy1 = new Proxy(sum, handler);

console.log(sum(1, 2)); // 3
console.log(proxy1(1, 2)); // 30
```

```js
function delay(f, ms) {
  return new Proxy(f, {
    apply(target, thisArg, args) {
      setTimeout(() => target.apply(thisArgs, args), ms);
    },
  });
}
```

Плюс данного метода в отличие от обертки, все обращение к функции будет на прямую, а не к функции обертке

# construct

ловушка для [[Construct]], метод который будет срабатывать с вызовом new

```js
function monster1(disposition) {
  this.disposition = disposition;
}

const handler1 = {
  construct(target, argumentsList, newTarget) {
    console.log(`Creating a ${target.name}`);
    // "Creating a monster1"

    return new target(...argumentsList);
  },
};

const proxy1 = new Proxy(monster1, handler1);

console.log(new proxy1("fierce").disposition);
```

## constructor И apply

```js
function extend(sup, base) {
  var descriptor = Object.getOwnPropertyDescriptor(
    base.prototype,
    "constructor"
  );

  const prototype = { ...base.prototype };

  base.prototype = Object.create(sup.prototype);
  base.prototype = Object.assign(base.prototype, prototype);

  var handler = {
    construct: function (target, args) {
      var obj = Object.create(base.prototype);
      this.apply(target, obj, args);
      return obj;
    },
    apply: function (target, that, args) {
      sup.apply(that, args);
      base.apply(that, args);
    },
  };
  var proxy = new Proxy(base, handler);
  descriptor.value = proxy;
  Object.defineProperty(base.prototype, "constructor", descriptor);
  return proxy;
}

var Person = function (name) {
  this.name = name;
};

var Boy = extend(Person, function (name, age) {
  this.age = age;
});

Boy.prototype.sex = "M";

var Peter = new Boy("Peter", 13);
console.log(Peter.sex); // "M"
console.log(Peter.name); // "Peter"
console.log(Peter.age); // 13
```

# defineProperty

задает новое свойство - возвращаемое свойство - игнорируется. Ловушка для [[DefineOwnProperty]]

defineProperty function(target, name, propertyDescriptor) -> any

- target
- name - свойство строковое или символьное
- propertyDescriptor - дескриптор
- - enumerable
- - configurable
- - writable
- - value
- - get
- - set

```js
const handler1 = {
  defineProperty(target, key, descriptor) {
    invariant(key, "define");
    return true;
  },
};

function invariant(key, action) {
  if (key[0] === "_") {
    throw new Error(`Invalid attempt to ${action} private "${key}" property`);
  }
}

const monster1 = {};
const proxy1 = new Proxy(monster1, handler1);

console.log((proxy1._secret = "easily scared"));
```

# deleteProperty

срабатывает на удаление, если возвращает true - успешно

deleteProperty function(target, name) -> boolean

```js
const monster1 = {
  texture: "scaly",
};

const handler1 = {
  deleteProperty(target, prop) {
    if (prop in target) {
      delete target[prop];
      console.log(`property removed: ${prop}`);
      // "property removed: texture"
    }
  },
};

console.log(monster1.texture); // "scaly"

const proxy1 = new Proxy(monster1, handler1);
delete proxy1.texture;

console.log(monster1.texture); // undefined
```

# get

срабатывает при proxy[foo] and proxy.bar

handler должен иметь метод get(target, property, receiver):

- target – оригинальный объект, который передавался перовым аргументом в конструктор new Proxy
- property – имя свойства всегда строковое значение, при массивах использовать +prop
- receiver – если свойство объекта является геттером, то receiver – это объект, который будет использован как this при его вызове. Обычно это сам объект прокси

```js
//  Массив, при чтение из которого несуществующего элемента возвращается 0 (обычно undefined)
let numbers = [0, 1, 2];

let numbers = new Proxy(numbers, {
  get(target, prop) {
    if (prop in target) {
      return target[prop];
    } else {
      return 0;
    }
  },
});

alert(numbers[1]); //1
alert(numbers[123]); //0 нет такого элемента
```

```js
//Словарь, в котором при неизвестном запросе возвращается фраза
let dictionary = {
  Hello: "Hola",
  Bye: "Adios",
};

let dictionary = new Proxy(dictionary, {
  //если прокси должен заменить оригинальный объект, то никто не должен  ссылаться на оригинал
  get(target, phrase) {
    if (phrase in target) {
      return target[phrase];
    } else {
      return phrase;
    }
  },
});
alert(dictionary["Hello"]); //Hola
alert(dictionary["Welcome to proxy"]); //Welcome to proxy
```

# getOwnPropertyDescription

ловушка для [[GetOwnProperty]]

getOwnPropertyDescriptor function(target, name) -> PropertyDescriptor | undefined

```js
const monster1 = {
  eyeCount: 4,
};

const handler1 = {
  getOwnPropertyDescriptor(target, prop) {
    console.log(`called: ${prop}`);
    // "called: eyeCount"

    return { configurable: true, enumerable: true, value: 5 };
  },
};

const proxy1 = new Proxy(monster1, handler1);

console.log(Object.getOwnPropertyDescriptor(proxy1, "eyeCount").value);
```

Пробросит ошибку:

- если результат не null или Object
- если configurable: false должно возвращать undefined
- если для существующего

# getPrototypeOf

ловушка для [[GetPrototypeOf]]

```js
const monster1 = {
  eyeCount: 4,
};

const monsterPrototype = {
  eyeCount: 2,
};

const handler = {
  getPrototypeOf(target) {
    return monsterPrototype;
  },
};

const proxy1 = new Proxy(monster1, handler);

console.log(Object.getPrototypeOf(proxy1) === monsterPrototype); // true

console.log(Object.getPrototypeOf(proxy1).eyeCount); // 2
```

Пробросит ошибку:

- если результат не null или Object

# has

ловушка для [[HasProperty]] при проверке prop in obj

has function(target, name) -> boolean

# isExtensible

Ловушка лоя [[IsExtensible]]

```js
const monster1 = {
  canEvolve: true,
};

const handler1 = {
  isExtensible(target) {
    return Reflect.isExtensible(target);
  },
  preventExtensions(target) {
    target.canEvolve = false;
    return Reflect.preventExtensions(target);
  },
};

const proxy1 = new Proxy(monster1, handler1);

console.log(Object.isExtensible(proxy1)); // true

console.log(monster1.canEvolve); // true

Object.preventExtensions(proxy1);

console.log(Object.isExtensible(proxy1)); // false

console.log(monster1.canEvolve); // false
```

# ownKeys

ловушка для [[OwnPropertyKeys]]

возвращает массив всех собственных имен [string | symbol]

Object.keys, цикл for...in и большинство других методов, которые работают со списком свойства объекта, используют внутренний метод [[OwnPropertyKeys]] (перехватываемый ловушкой ownKeys) для их получения

```js
let user = { name: "Вася", age: 30, _password: "***" };

user = new Proxy(user, {
  ownKeys(target) {
    return Object.keys(target).filter((key) => !key.startWith("_"));
  },
});

for (let key in user) alert(key); //name, age
alert(Object.keys(user)); //name, age
alert(Object.values(user)); //Вася, 30

let user = {};
user = new Proxy(user, {
  ownKeys(target) {
    return ["a", "b", "c"];
  },
});
alert(Object.keys(user)); //пусто если мы попробуем возвратить ключ, которого на самом деле нет, то  Object.keys его не выдаст, пропущен так как нет флага enumerable
```

Чтобы возвращалось свойство

```js
let user = {};
user = new Proxy(user, {
  ownKeys(target) {
    return ["a", "b", "c"];
  },

  getOwnPropertyDescriptor(target, prop) {
    return {
      enumerable: true,
      configurable: true,
    };
  },
});

alert(Object.keys(user)); //a, b, c
```

# preventExtensions

ловушка [[PreventExtensions]]

preventExtensions function(target) -> boolean

```js
const monster1 = {
  canEvolve: true,
};

const handler1 = {
  preventExtensions(target) {
    target.canEvolve = false;
    Object.preventExtensions(target);
    return true;
  },
};

const proxy1 = new Proxy(monster1, handler1);

console.log(monster1.canEvolve); // true

Object.preventExtensions(proxy1);

console.log(monster1.canEvolve); // false
```

# set

set(target, property, value, receiver):

- target – оригинальный объект, который передавался первым аргументом в конструктор new Proxy property – имя свойства
- value – значение свойства
- receiver – этот аргумент имеете значение, если только свойство сеттер. Set должна вернуть true если запись прошла успешно и false в противном случае(будет сгенерирована ошибка TypeError)
  Массив исключительно для чисел

```js
let numbers = [];
numbers = new Proxy(numbers, {
  set(target, prop, val) {
    if (typeof val === "number") {
      target[prop] = val;
      return true;
    } else {
      return false;
    }
  },
});

numbers.push(1); //добавлен
numbers.push(2); //добавлен
numbers.push("Тест"); //TypeError
```

# setPrototypeOf()

ловушка дял [[SetPrototypeOf]]

```js
const handler1 = {
  setPrototypeOf(monster1, monsterProto) {
    monster1.geneticallyModified = true;
    return false;
  },
};

const monsterProto = {};
const monster1 = {
  geneticallyModified: false,
};

const proxy1 = new Proxy(monster1, handler1);
// Object.setPrototypeOf(proxy1, monsterProto); // Throws a TypeError

console.log(Reflect.setPrototypeOf(proxy1, monsterProto));
// Expected output: false

console.log(monster1.geneticallyModified);
// Expected output: true
```

<!--  -->

# Proxy.revocable()

отзыв обработчика

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs:

## BP. защищенные свойства с ловушкой

```js
let user = { name: "Вася", _password: "Secret" };
let user = new Proxy(user, {
  get(target, prop) {
    if (prop.startWith("_")) {
      throw new Error("Отказано в доступе");
    } else {
      let value = target[prop];
      return typeof value === "function" ? value.bind(target) : value; // метод объекта должен  иметь доступ к _password
    }
  },
  set(target, prop, val) {
    if (prop.startWith("_")) {
      throw new Error("Отказан в доступе");
    } else {
      target[prop] = val;
      return true;
    }
  },
  deleteProperty(target, prop) {
    if (prop.startWith("_")) {
      throw new Error("Отказано в доступе");
    } else {
      delete target[prop];
      return true;
    }
  },
  ownKeys(target) {
    return Object.keys(target).filter((key) => !key.startWith("_"));
  },
});

try {
  alert(user.password);
} catch (e) {
  alert(e.message);
}

try {
  user.password = "test";
} catch (e) {
  alert(e.message);
}

try {
  delete user._password;
} catch (e) {
  alert(e.message);
}

for (let key in user) {
  alert(key);
}
```

## в диапазоне с ловушкой has

```js
let range = { start: 1, end: 10 };
```

Сделать оператор in чтобы проверить, что некоторое число находится в указанном диапазоне has(target, property)//target – это оригинальный объект, который был первым аргументом в new Proxy property – имя свойства

```js
let range = {
  start: 1,
  end: 10,
};

range = new Proxy(range, {
  has(target, prop) {
    return prop >= target.start && prop <= target.end; //true/false значение
  },
});

alert(5 in range); //true
alert(50 in range); //false
```

## BP

```js
// 14.1 Задача №3. Observable

let handlers = Symbol("handlers"); //создается символ handler с описанием handlers
function makeObservable(target) {
  target[handlers] = []; //создаем в обрабатываемом объекте символьное свойство handlers с ключом в виде  пустого массива, для каждого следующего оборачиваемого объекта будет свое индивидуальное, символьное  свойство с именем handlers

  target.observe = function (handler) {
    // в объекте создается метод observe с аргументом в виде handler
    this[handlers].push(handler); //this == target (user в этом случае), в массив добавляется handler
  };

  return new Proxy(target, {
    //создание Proxy
    set(target, property, value, receiver) {
      let success = Reflect.set(...arguments); //set возвращает success если выполнен без ошибок
      if (success) {
        target[handlers].forEach((handler) => handler(property, value)); //для каждого элемента массива в виде методов вызываем обработчики с аргументами property и value
      }
      return success;
    },
  });
}

let user = {};
user = makeObservable(user);

user.observe((key, value) => alert(`SET ${key} = ${value}`));
//эта стрелочная функция будет аргументом handler

user.name = "John";
```
