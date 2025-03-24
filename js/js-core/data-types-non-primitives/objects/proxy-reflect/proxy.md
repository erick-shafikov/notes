# Proxy

- прокси-объект - обертка над объектом целью
- обработчики - объект с ловушками
- ловушки - методы переопределяющий поведение объекта
- цель - исходный объект

Объект proxy оборачивается вокруг объекта и может перехватывать разные действия с ним. Синтаксис:

```js
//target – объект, для которого нужно Proxy
let proxy = new Proxy(target, {
  //handler – конфигурация proxy: объект с ловушками, методами, которые перехватывают разные операции. Proxy без ловушек:
  getPrototypeOf() {
    //Object.getPrototypeOf(), Reflect.getPrototypeOf(), __proto__, Object.prototype.isPrototypeOf(), instanceof
  },
  setPrototypeOf() {
    //Object.setPrototypeOf() и Reflect.setPrototypeOf()
  },

  isExtensible() {
    //Object.isExtensible() Reflect.isExtensible()
  },
  preventExtensions() {
    //Object.preventExtensions(), Reflect.preventExtensions()
  },
  getPropertyDescriptor: function (oTarget, sKey) {
    var vValue = oTarget[sKey] || oTarget.getItem(sKey);
    return vValue
      ? {
          value: vValue,
          writable: true,
          enumerable: true,
          configurable: false,
        }
      : undefined;
  },
  getOwnPropertyDescriptor(oTarget, sKey) {
    var vValue = oTarget[sKey] || oTarget.getItem(sKey);
    return vValue
      ? {
          value: vValue,
          writable: true,
          enumerable: true,
          configurable: false,
        }
      : undefined; //или undefined
    //Object.getOwnPropertyDescriptor(), Reflect.getOwnPropertyDescriptor()
  },
  defineProperty(oTarget, sKey, oDesc) {
    if (oDesc && "value" in oDesc) {
      oTarget.setItem(sKey, oDesc.value);
    }
    return oTarget;
    //Object.defineProperty(), Reflect.defineProperty(),
  },
  has(oTarget, sKey) {
    return sKey in oTarget || oTarget.hasItem(sKey);
    //foo in proxy, foo in Object.create(proxy), Reflect.has()
  },
  get(oTarget, sKey) {
    return oTarget[sKey] || oTarget.getItem(sKey) || undefined;
    //proxy[foo]and proxy.bar, Object.create(proxy)[foo], Reflect.get(),
  },
  set(oTarget, sKey, vValue) {
    if (sKey in oTarget) {
      return false;
    }
    return oTarget.setItem(sKey, vValue);
    //roxy[foo] = bar and proxy.foo = bar, Reflect.set()
  },
  deleteProperty(oTarget, sKey) {
    if (sKey in oTarget) {
      return false;
    }
    return oTarget.removeItem(sKey);
    //delete proxy[foo] and delete proxy.foo, Reflect.deleteProperty();
  },
  enumerate(oTarget, sKey) {
    return oTarget.keys();
  },
  iterate(oTarget, sKey) {
    return oTarget.keys();
  },
  ownKeys(oTarget, sKey) {
    return oTarget.keys();
  },
  hasOwn(oTarget, sKey) {
    return oTarget.hasItem(sKey);
  },
  getPropertyNames(oTarget) {
    return Object.getPropertyNames(oTarget).concat(oTarget.keys());
  },
  getOwnPropertyNames(oTarget) {
    return Object.getOwnPropertyNames(oTarget).concat(oTarget.keys());
  },
  fix: function (oTarget) {
    return "not implemented yet!";
  },
});
```

```js
let target = {};
let proxy = new Proxy(target, {}); //пустой handler

proxy.test = 5; //записали в proxy

alert(proxy.test); //5

for (let key in proxy) {
  alert(key); //test
}
```

# get

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

# set

set(target, property, value, receiver):

- target – оригинальный объект, который передавался первым аргументом в конструктор new Proxy property – имя свойства
- value – значение свойства
- receiver – этот аргумент имеете значение, если только свойство сеттер. Set должна вернуть true если запись прошла успешно и false в противном случае(будет сгенерирована ошибка TypeError)
  Массив исключительно для чисел

```js
let numbers = [];
let numbers = new Proxy(numbers, {
  set(target, prop, val){
    if (typeof val === "number"){
      target[prop] = val;
      return true; //при успешном добавлении не забывать возвращать true
      } else {
        return false;
        }}}
);
numbers.push(1);//добавлен
numbers.push(2);//добавлен
numbers.push("Тест")://TypeError

```

## getOwnPropertyDescription

getOwnPropertyDescriptor function(target, name) -> PropertyDescriptor | undefined

## ownKeys

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

## defineProperty

задает новое свойство - возвращаемое свойство - игнорируется

defineProperty function(target, name, propertyDescriptor) -> any

## deleteProperty

срабатывает на удаление, если возвращает true - успешно

deleteProperty function(target, name) -> boolean

## preventExtensions

preventExtensions function(target) -> boolean

## has

has function(target, name) -> boolean

<!--  -->

# Proxy.revocable()

отзыв обработчика

# constructor И apply

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

## BP. в диапазоне с ловушкой has

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

# Оборачивание функций

- apply(target, thiArg, args) – активируется при вызови прокси функции
- target – это оригинальный объект
- thisArg – это контекст this
- args – список аргументов

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
