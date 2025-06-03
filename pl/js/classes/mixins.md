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
