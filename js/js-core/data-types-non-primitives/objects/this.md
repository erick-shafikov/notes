<!-- this ------------------------------------------------------------------------------------------------------------------------------------>

# this

```js
// эти объекты делают одно и то же (одинаковые методы)
user = {
  sayHi: function () {
    alert("Привет");
  },
};
// сокращённая запись
user = {
  sayHi() {
    // то же самое, что и "sayHi:  function()"
    alert("Привет");
  },
};

// this- это объект перед точкой, которой  использовался для вызова метода

let user = {
  name: "John",
  age: "30",
  sayHi() {
    alert(this.name);
  },
};
user.sayHi(); //John

// this не является фиксированным, его значение  зависит от контекста
let user = { name: "John" };
let admin = { name: "Admin" };
function sayHi() {
  alert(this.name);
}
user.f = sayHi;
admin.f = sayHi;
user.f(); //John
admin.f(); //Admin
// вызов без объекта this == undefined
function sayHi() {
  alert(this);
}
sayHi(); //undefined в данном случае ссылка на глобальный объект window
```

## Потеря this

```js
let user = {
  firstName: "Vasya",
  setHi() {
    alert(`Hi, ${this.name}!`);
  },
};
setTimeout(user.sayHi, 1000); // hi, undefined, так как setTimeout устанавливает this = window

//Решение 1 функция – обертка объект тот же
setTimeout(function () {
  user.sayHi();
}, 1000);
```

есть уязвимость – если до момента вызова setTimeout user измениться, то setTimeout вызовет его с изменениям

# Reference type

```js
let user = {
  name: "John",
  hi() {
    alert(this.name);
  },
  bye() {
    alert("bye");
  },
};
user.hi(); // работает

(user.name == "John" ? user.hi : user.bye)(); //ошибка
```

(object.method())
операция – оператор точка object.method() – возвращает свойство объекта – его метод obj.method
операция – скобки () вызывают этот метод

если мы переместим операции в отдельные строки , то значение будет потерянно

```js
let user = {
  name: "John",
  hi() {
    alert(this.name);
  },
};

let hi = user.hi; //сохраняет функцию в переменной здесь нет this
hi(); //Ошибка так как значение this == undefined
```

user.hi() – точка возвращает не саму функцию, а специальное значение ссылочного типа (base, name, strict)
base – объект, name – свойства объекта, в нашем случае будет (user, "hi", true), когда () применяются к значению ссылочного типа, то ставится правильный this, при любых других операциях ссылочный тип заменяется на значение user.hi, то есть правильный this будет в случае

Сам по себе вызов (obj.method)() – кардинально ни чем не отличаются

```js
function makeUser() {
  //здесь this == undefined
  return {
    name: "John",
    ref: this,
  };
}
let user = makeUser();
alert(user.ref.name); //Ошибка Cannot read property "name" or undefined
```

```js
let user = {
  firstName: "John",
  seyHi() {
    let arrow = () => alert(this.firstName);
    arrow();
  },
};
user.seyHi(); //John
```

Правила определения this никак не смотрят на объявление объекта this внутри makeUser() == undefined, литерал не важен, таким образом ref: this берет this из makeUser()

```js
function makeUser() {
  return {
    name: "John",
    ref() {
      return this;
    },
  };
}
let user = makeUser();
alert(user.ref().name); //John – теперь user.ref() вызывается как метод и значение this будет объект перед точкой
```

```js
// TG;
"use strict";
const obj = {
  foo() {
    //console.log(this.name) Marco Polo
    function getName() {
      console.log(this.name);
    }
    getName();
  },
  name: "Marco Polo",
  callbackFoo(callback) {
    callback();
  },
};
obj.foo(); //ошибка в use strict режиме, а без use strict - пустая строка
obj.callbackFoo(function () {
  console.log(this); //undefined в use strict, без use strict объект window
});
// TG;
window.a = 777;
function f() {
  console.log(this.a);
}
const g = () => console.log(this.a);
const obj = { a: 666, f, g };
obj.f(); //666
obj.g(); //777 если window.a = 777 или var a = 777, если поменять на let то undefined
```
