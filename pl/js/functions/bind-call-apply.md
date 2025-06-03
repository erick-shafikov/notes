# bind

```js
// Синтаксис:

let boundFunc = func.bind(context);

let user = {
firstName: "Вася"
};

function func(){
  alert(this.firstName);
}
let funcUser = func.bind(user);
funcUser(); //Вася

let user = {firstName: "Вася"};

function func(phrase){
  alert(phrase + "," + this.firsName);
}

let funcUser = func.bind(user):
funcUser("Привет");

// С методом объекта
let user = {  firstName: "Вася",  sayHi(){
alert(`Привет ${this.firstName}`);
}
};
let sayHi = user.sayHi.bind(user);// setTimeout(sayHi, 1000), не потерялось, даже в силу
```

Если у объекта много методов

```js
for (let key in user) {
  if (typeof user[key] == "function") {
    user[key] = user[key].bind(user); // в lodash _.bindAll(obj)
  }
}
```

полный синтаксис

```js
let bound = func.bind(context, arg1, arg2); //позволяет привязать не только контекст, но и аргументы  функции

function mul(a, b) {
  return a * b;
}

let double = mul.bind(null, 2); //Функция будет удваивать значение переданного нового аргумента
//null фиксируется как контекст, 2 как первый аргумент, мы не используем this, но для bind это обязателный
//аргумент, triple = mul.bind(null, 3) выглядел бы так
```

```js
var a = {
  name: "a",
  foo: function () {
    console.log(this.name);
  },
};
var b = { name: "b" };
var c = { nsme: "c" };
a.foo.bind(b).bind(c)(); //b так как контекст можно привязать только один раз
```

# func.apply()

func.apply();
func.apply(context, args) разница между call и apply состоит в том, что call ожидает список аргументов, в то время как apply принимает псевдо массив
func.call(context, …args)//Оператор расширения… позволяет передавать перебираемый объект args в Виде
списка в call
func.call(context, …arg) – принимает массив как список с оператором расширения func.apply(context, args ) – тот же эффект (лучше)
Передача всех аргументов вместе с контекстом другой функции называется перенаправлением вызова call
forwarding
Пример разницы между call и apply

```js
let wrapper = function () {
  return func.apply(this, arguments);
};

const obj1 = {
  result: 0,
};
const obj2 = {
  result: 0,
};
function reduceAdd() {
  let result = 0;
  for (let i = 0, len = arguments.length; i < len; i++) {
    result += arguments[i];
  }
  this.result = result;
}
reduceAdd.apply(obj1, [1, 2, 3, 4, 5]);
reduceAdd.call(obj2, 1, 2, 3, 4, 5);
console.log(obj1.result); //15
console.log(obj2.result); //15
```

# func.call()

```js
// Синтаксис:
func.call(context, arg1 , arg2) //первый аргумент – ее контекст  func(1, 2, 3);
func.call(obj, 1, 2, 3); вызывает func с this = obj

// Пример:
function() {
alert(this.name);
}

let user = { name: "John" };
let admin = { name: "Admin" };

sayHi.call( user ); //John
sayHi.call( admin ); //Admin

```

```js
let worker = {
  someMethod() {
    return 1;
  },
  slow(x) {
    alert("called with" + x);
    return x * someMethod();
  },
};
```

```js
function cachingDecorator(func) {
  let cache = newMap();
  return function (x) {
    if (cache.has(x)) {
      return cache.get(x);
    }
    let result = func.call(x);
    cache.set(x, result);
    return result;
  };
}
alert(worker.slow(1)); // оригинальный метод работает
worker.slow = cachingDecorator(worker.slow);
alert(worker.slow(2)); // ошибка

// тоже самое что и
let func = worker.slow; // откуда функция знает значение this  func(2);
```

```js
// случай с двумя параметрами:
let worker = {
  slow(min, max){
    return min+max
};
}

function cachingDecorator(func, hash){
  let cache = newMap();

return function(){
let key = hash(arguments);//arguments текущей функции, передаются и склеиваются
if cache.has(key){
return cache.get(key);
}

let result = func.call(this,...arguments);//передаем все аргументы функции-исходника  cache.set(key, result);
return result;
};
}

function hash(args){
return args[0]+","args[1]
}

worker.slow = cachingDecorator(worker.slow, hash);

// вместо func.call можно использовать func.apply

```

```js
function myFunc() {
  alert(this);
}
myFunc.call(null); //object Window в случает сели передается null в call, то this === [object Windiow]
TG;
function f(a, b, c) {
  const s = Array.prototype.join.call(arguments);
  console.log(s);
}
f(1, "a", true); //1,'a',true – возвращает строку через запятую
```

# Частичное применение

```js
// Частичное применение без контекста
// если мы хотим зафиксировать некоторые аргументы, но не контекст this
function partial(func, ...argBounds) {
  return function (...args) {
    return func.call(this, ...argsBounds, ...args);
  };
}

let user = {
  firstName: "John",
  say(time, phrase) {
    alert(`[${time}] ${this.firstName}: ${phrase}!`);
  },
};

user.sayNow = partial(
  user.say,
  new Date().getHours() + ":" + new Date().getMinutes()
);
user.sayNow("Hello"); //[10:00] John: Hello!  this	== user, полученный из user.sayNow
// передаем argBounds – аргументы из вызова partial("10:00")  затем передаем ...args – аргументы полученный оберткой ("Hello")
```

# заимствование метода

```js
function hash(args) {
  // работает только для двух аргументов  return args[0] + "," + args[1];
}

// для нескольких аргументов:
function hash(args) {
  return args.join(); // не сработает объект arguments является перебираемым и псевдо массивом, а не реальным массивом
}

function hash() {
  return [].join.call(arguments); // теперь сработает
}

// так как join работает с this[0], this[1] то с this = arguments
```
