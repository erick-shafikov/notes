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
