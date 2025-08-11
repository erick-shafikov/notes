# Reflect

Reflect – предоставляет объект с методами, которые позволяют выполнять стандартные действия над объектом

- возвращают true/false если операция прошла успешно

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

# apply

позволяет вызывать функцию с аргументами

```js
console.log(Reflect.apply(Math.floor, undefined, [1.75])); // 1

console.log(
  Reflect.apply(String.fromCharCode, undefined, [104, 101, 108, 108, 111])
); // "hello"

console.log(
  Reflect.apply(RegExp.prototype.exec, /ab/, ["confabulation"]).index
); // 4

console.log(Reflect.apply("".charAt, "ponies", [3])); // "i"
```

# construct

работает как new оператор

Reflect.defineProperty()
Reflect.deleteProperty()
Reflect.get()
Reflect.getOwnPropertyDescriptor()
Reflect.getPrototypeOf()
Reflect.has()
Reflect.isExtensible()
Reflect.ownKeys()
Reflect.preventExtensions()
Reflect.set()
Reflect.setPrototypeOf()

# get

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
```

```js
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

let { proxy, revoke } = Proxy.revocable(target, handler);

let object = {
  data: "Важные Данные",
};

let { proxy, revoke } = Proxy.revocable(object, {});
alert(proxy.data);
revoke();
alert(proxy.data); //ошибка
```

# observable на proxy и reflect

```js
// Хранилище подписчиков
const subscribers = new Map();

// Функция добавления подписчика на свойство
function subscribe(property, callback) {
  if (!subscribers.has(property)) {
    subscribers.set(property, []);
  }
  subscribers.get(property).push(callback);
}

const reactiveHandler = {
  get(target, prop, receiver) {
    console.log(`Чтение свойства "${prop}"`);
    // Используем Reflect для делегирования стандартной операции get
    return Reflect.get(target, prop, receiver);
  },
  set(target, prop, value, receiver) {
    console.log(`Изменение свойства "${prop}" на "${value}"`);
    const result = Reflect.set(target, prop, value, receiver);
    if (result && subscribers.has(prop)) {
      // Уведомляем подписчиков об изменении свойства
      subscribers.get(prop).forEach((callback) => callback(value));
    }
    return result;
  },
};

const data = {
  name: "Иван",
  age: 30,
};

// Создаем Proxy
const reactiveData = new Proxy(data, reactiveHandler);

// Подписываемся на изменения свойства name
subscribe("name", (newValue) => {
  console.log(`Имя изменилось на: ${newValue}`);
});

console.log(reactiveData.name);
reactiveData.name = "Алексей";
```
