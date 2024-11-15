# Proxy

Объект proxy оборачивается вокруг объекта и может перехватывать разные действия с ним. Синтаксис:

```js
let proxy = new Proxy(target, handler); //target – объект, для которого нужно Proxy
//handler – конфигурация proxy: объект с ловушками, методами, которые перехватывают разные операции. Proxy без ловушек:
```

```js
let target = {};
let proxy = new Proxy(target, {}); //пустой handler
proxy.test = 5; //записали в proxy
alert(proxy.test); //5
for (let key in proxy) alert(key); //test
```

![alt text](/assets/js/proxy-table.png)

## Значение по умолчанию с ловушкой get

handler должен иметь метод get(target, property, receiver)
target – оригинальный объект, который передавался перовым аргументом в конструктор new Proxy property – имя свойства //всегда строковое значение, при массивах использовать +prop
receiver – если свойство объекта является геттером, то receiver – это объект, который будет использован
как this при его вызове. Обычно это сам объект прокси
Массив, при чтение из которого несуществующего элемента возвращается 0 (обычно undefined)

```js
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
alert(numbers[123]); //0 нет такого элемента Словарь, в котором при неизвестном запросе возвращается фраза
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

## Валидация с ловушкой get

set(target, property, value, receiver)
target – оригинальный объект, который передавался первым аргументом в конструктор new Proxy property – имя свойства
value – значение свойства
receiver – этот аргумент имеете значение, если только свойство сеттер. Set должна вернуть true если запись прошла успешно и false в противном случае(будет сгенерирована ошибка TypeError)
Массив исключительно для чисел

```js
let numbers = [];
let numbers = new Proxy(numbers, {
  set(target, prop, val){
if (typeof val == "number"){
target[prop] = val;
return true;//при успешном добавлении не забывать возвращать true
} else {
  return false;
}}});
numbers.push(1);//добавлен
numbers.push(2);//добавлен
numbers.push("Тест")://TypeError

```

## перебор с помощью ownKeys и getOwnPropertyDescription

Object.keys, цикл for…in и большинство других методов, которые работают со списком свойства объекта, используют внутренний метод [[OWnProertyKeys]] (перехватываемый ловушкой ownKeys) для их получения

Object.getOwnPropertyNames(obj) возвращает не-символьные ключи Object.getOwnPropertySymbols(obj) возвращает символьные ключи Object.keys/values() возвращает не-символьные ключи/значения с флагом enumerable
for…in перебирает не-символьные ключи с флагом enumerable, а также ключи прототипов пропускаем свойства начинающиеся с подчеркивания \_:

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

## защищенные свойства с ловушкой

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

for (let key in user) alert(key);
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

## Оборачивание функций

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

# Reflect

Reflect – встроенный объект упрощающий создание прокси, позволяющий обернуть методы и правильно перенаправить

```js
let user = {
  name: "Вася",
};

user = new Proxy(user, {
  get(target, prop, receiver) {
    alert("GET ${prop}");
    return Reflect.get(target, prop, receiver);
  },

  set(target, prop, val, receiver) {
    alert(`SET ${prop} = {val}`);
    return Reflect.set(target, prop, val, receiver);
  },
});

let name = user.name;
user.name = "Петя";
```

## Прокси для геттера

Раскроем суть receiver

```js
let user = {
  _name: "Гость",
  get name() {
    return this._name;
  },
};

let userProxy = new Proxy(user, {
  get(target, prop, receiver) {
    return target[prop]; //target == user
  },
});

alert(userProxy.name);

let admin = {
  proto: userProxy,
  _name: "Admin",
};

alert(admin.name); //Гость

let userProxy = new Proxy(user, {
  get(target, prop, receiver) {
    return Reflect.get(target, prop, receiver);
  },

  // Можно было переписать как
  get(target, prop, receiver) {
    return Reflect.get(...arguments);
  },
});
```

### Ограничения для прокси

Map, Set, Date, Promise – имеют внутренние слоты
Map хранит элементы во внутреннем слоте [[MapData]], Встроенные методы get/ set обращаются напрямую, не через get/set

```js
let map = new Map();
let proxy = new Proxy(map, {});
proxy.set("test", 1);

// исправим

let proxy = new Map();
let proxy = new Proxy(map, {
  get(target, prop, receiver) {
    let value = Reflect.get(...arguments);
    return typeof value == "function" ? value.bind(target) : value;
  },
});

proxy.set("test", 1);
alert(proxy.get("test")); //1

// ----------------------------------------------------------------------
class User {
  #name = "Гость";

  getName() {
    return this.#name;
  }
}

let user = new User();

user = new Proxy(user, {});
alert(user.getName()); //Ошибка

// исправление:

let user = new Proxy(user, {
  get(target, prop, receiver) {
    let value = Reflect.get(...arguments);
    return typeof value == "function" ? value.bind(target) : value;
  },
});

alert(user.getName());


let proxy {proxy, revoke} = Proxy.revocable(target, handler)

let object = {
data: "Важные Данные",
};

let {proxy, revoke} = Proxy.revocable(object, {});
alert(proxy.data);
revoke()
alert(proxy.data)//ошибка

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

user.observe((key, value) => alert("SET ${key} = ${value}"));
//эта стрелочная функция будет аргументом handler

user.name = "John";
```
