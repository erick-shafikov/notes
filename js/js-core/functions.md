# Функции

!!! Переменные объявленные внутри функции видны только внутри функции
!!! Может изменять значение внешних переменных, если внутри нет такой же переменной, меняет их после вызова функции , если внутри есть такая переменная, то не изменяет внешнюю.
!!! Function declaration, если находятся внутри блоков {}, вне их не видны, обойти можно про использовании FE
!!!Параметры по умолчанию undefined
!!!В параметрах может быть функция

```js
function showMessage(from, text = "текст не добавлен") {
  alert(from + ": " + text);
}
showMessage("Аня"); // Аня: текст не добавлен
```

# Add-hoc

Когда пользователь отвечает на вопрос – функция называется обработчиком. Можем передать 2 типа обработчиков – функцию без аргументов для положительного ответа и функцию с аргументами, которая будет вызываться в любом случае

```js
function ask(question, …handlers){ // вопросы и обработчик
let isYes = confirm(question); // isYes принимает либо true либо false

for (let handler of handlers){ //для каждого из handlers
if (handler.length == 0){ //если количество обработчиков = 0, то есть их нет, в этом примере не  выполнится эта часть
if (isYes) handler(); //и если ответ true для isYes то вызывается, если false, то нет
} else { // если handler не пустой, то передать true, как аргумент в каждый объект, 2 раза  выполнится эта часть, так как
handler(isYes);
}}}
ask("Вопрос?", ()=> alert("Вы ответили да"), result => alert(result)); // в любом случае вызывается  вторая функция.

```

# Arrow. Функции стрелки

```js
// создаёт функцию func с аргументами arg1..argN и вычисляет  expression с правой стороны с их использованием, возвращая  результат:
let func = (arg1, arg2, ...argN) => expression; // Пример:
let age = prompt("Сколько Вам лет?", 18);
let welcome = age < 18 ? () => alert("Привет") : () => alert("Здравствуйте!");
welcome(); // Если у нас только один аргумент, то круглые скобки вокруг  параметров можно опустить, сделав запись ещё короче:
// тоже что и
// let double = function(n) { return n * 2 }  let double = n => n * 2;
alert(double(3)); // 6 // Если нет аргументов, указываются пустые круглые скобки:
let sayHi = () => alert("Hello!"); // Если несколько инструкций, то заключить в скобки {}
let sayBye = () => {
  alert("Hello!");
  alert("Bye!");
};
// у стрелочных функций нет this, если идет обращение  к this, То его значение берется снаружи
let group = {
  title: "Our Group",
  students: ["john", "pete", "alice"],
  showList() {
    this.students.forEach((student) => alert(this.title + ":" + student));
  },
};
group.showList(); // Стрелочные функции не имеют "arguments"
function defer(f, ms) {
  return function () {
    setTimeout(() => f.apply(this, arguments), ms);
  };
}
function sayHI(who) {
  alert(`Hello, ` + who);
}
let sayHiDeferred = defer(sayHi, 2000);
sayHiDeferred("John");

// TG повторное объявление после параметра
var a = 1;
function foo(x = 2) {
  let x = 5;
  console.log(x);
}
foo(); // Uncaught SyntaxError: Identifier 'x' has already been declared
```

Нет свойства constructor

# bind, call, apply

## bind

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

## Частичное применение

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

## Частичное применение без контекста

у стрелочных функций нет this

```js
let group = {
  title: "Our Group",
  students: ["JOhn", "Pete", "Alice"],
  showList() {
    // работает this.title = group.title
    this.students.forEach((students) => alert(this.title + ":" + student));
  },

  showList() {
    // не сработает title = undefined,
    // так как forEach выполняет с this = undefined
    this.students.forEach(function (student) {
      alert(this.title + ":" + student);
    });
  },
};

group.showList();
```

!!!Нельзя использовать с new
!!!При использовании с bind, как обычная переменная берется из внешнего ЛО

## func.apply()

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

## func.call()

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

## заимствование метода

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

## Декораторы

```js
function slow(x) {
  alert(`called with ${x}`);
  return x;
}

function cachingDecorator(func) {
  let cache = new Map(); //создаем Map
  return function (x) {
    //возвращаем функцию с  аргументом х
    if (cache.has(x)) {
      //если есть результат, то возвращаем просто результат
      return cache.get(x);
    }
    let result = func(x); //если  результата нет, то запоминаем его
    cache.set(x, result);
    return result;
  };
}

slow = cachingDecorator(slow);
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
function work(a, b) {
  alert(a + b); // произвольная функция или метод
}
function spy(func) {
  wrapper.calls = [];
  //так как мы вернем функцию wrapper, то calls будет внутренним свойством с ключом в виде массива

  function wrapper(...args) {
    // фишка с подкидыванием …args для получения аргументов оборачиваемой функции в массив
    wrapper.calls.push(args);
    return func.apply(this, arguments);
    // если убрать <….apply(this…> – результат вычислений [object Arguments]undefined, так как мы возвращаем  wrapper, при обертывании теряется контекст и объект Arguments становится undefined
  }

  return wrapper;
}
work = spy(work);
work(1, 2); // 3
work(4, 5); // 9

for (let args of work.calls) {
  alert("call:" + args.join()); // "call:1,2", "call:4,5"
}
```

```js
function delay(f, ms) {
  return function () {
    setTimeout(() => f.apply(this, arguments), ms);
  };
}

let f1000 = delay(alert, 1000);
f1000("test"); // показывает "test" после 1000 мс  второй вариант

function delay(f, ms) {
  return function (...args) {
    let savedThis = this; // сохраняем this в промежуточную переменную

    setTimeout(function () {
      f.apply(savedThis, args); // используем её
    }, ms);
  };
}
```

```js
function defer(f, ms) {
  //функция defer откладывает вызов функции f на ms секунд
  return function () {
    setTimeout(() => f.apply(this, arguments), ms);
  };
}
function sayHi(who) {
  alert("Hello", +who);
}

let sayHiDeffer = defer(sayHi, 2000);

sayHiDeffer("John");

//без стрелочных функций
function defer(f, ms) {
  return function (...args) {
    let ctx = this; //создаем дополнительные переменные ctx и args чтобы функция внутри setTimeout могла получить их
    setTimeout(function () {
      return f.apply(ctx, args);
    }, ms);
  };
}
// TG;
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

# Caring

```js
function sum(a) {
  let tempSum = a; //текущая сумма аргументов
  function addSum(b) {
    // функция добавочного аргумента, которая прибавляет добавочный аргумент к ткущей сумме
    tempSum += b; //функция меняет значение tempSum
    return addSum; // функция возвращает себя для дальнейшего добавления аргументов, именно этот  шаг «заводит» функцию для добавления аргументов произвольного количества скобок, так как return
  }

  addSum.toString = function () {
    //метод для преобразования в строку, возвращает текущую сумму  return tempSum;
  };

  return addSum; // функция возвращает вложенную функцию добавочного аргумента, один этот return  сработал бы только для вторых скобок
}

alert(sum(1)(2));
alert(sum(1)(2)(3));
alert(sum(5)(-1)(2));
alert(sum(6)(-1)(-2)(-3));
alert(sum(0)(1)(2)(3)(4)(5));

// схема:
function f(a) {
  function g(b) {
    return g;
  }
  return g;
}
```

## Продвинутая реализация каррирования со множеством аргументов

Каррирование функции – трансформация функции, при которой function(a, b, c) может вызываться как function(a)(b)(c)
Каррирование не вызывает функцию, оно просто трансформирует ее Работает только с фиксированным количеством аргументов

```js
function curry(func) {
  return function curried(...args) {
    if (args.length >= func.length) {
      //если количество переданных аргументов args совпадает c количеством аргументов при объявлении функции func тогда функция переходит к ней и выполняет ее
      return func.apply(this, args);
    } else {
      return function (...args2) {
        //если аргументов в вызове меньше, ты вызывается обертка которая складывает вызовы и аргументы в args рекурсия
        return curried.apply(this, args.concat(args2));
      };
    }
  };
}
function sum(a, b, c) {
  return a + b + c;
}
let curriedSum = curry(sum);
alert(curriedSum(1, 2, 3));
alert(curriedSum(1)(2, 3));
alert(curriedSum(1)(2)(3));
```

# Closures

Контекст выполнения – execution context – структура данных, которая содержит информацию о вызове функции включает в себя: место в коде, где находится интерпретатор, локальные переменные, значение this.
Один вызов – один контекст выполнения
При вложенном вызове происходит:

- выполнение текущей функции останавливается
- Её контекст запоминается в стеке контекстов
- Выполняется каждый вложенный вызов, для каждого из которого создается свой контекст
- после завершения старый контекст достается из стека и выполнение функция возобновляется с того места, где она была остановлена

## LE

у каждой функции, блок и скрипта есть связанный с ним внутренний объект – Lexical Environment, который состоит из:

- Environment Record – объект в котором хранятся локальные переменные и такие вещи как this. Переменная в свою очередь – это свойство этого объекта ER. Получить или изменить переменную = получить или изменить свойство этого внутреннего объекта.
- Ссылка на внешнее логическое окружение – то, что находится за скобками { }

Переменная - это свойство внутреннего объекта Lexical Enviroment. Получить или изменить переменную, означает получить или изменить это свойство, с этим LE идет в паре ссылка outer либо на ноль , либо на другое LE

В процессе вызова функции есть два лексических окружения внутреннее (для вызываемой функции) и внешнее (глобальное). Если переменная не была найдена, то в strict modе это будет ошибкой, а без strict mode создается глобальная переменная.

один вызов – одно лексическое окружение

Все функции, при рождении получают свойство [[Environment]], которое ссылается на лексическое окружение
места, где он были созданы. Другими словами это свойство - ссылка на лексическое окружение

Замыкание – это функция, которая запоминает свои внешние переменные и может получить к ним доступ. Они автоматически запоминают место, где были созданы, с помощью свойства [[Environment]] и все они могут получить доступ к внешним переменным.

## блоки кода

```js
// if
let phrase = "hello";
if (true) {
  let user = "John";
  alert(`${phrase}${user}`);
}
alert(user); //Error, no such variable т.к. переменная user существует только в блоке кода if

// for, while
for (let i = 0; i < 10; i++) {
  //у каждой итерации цикла свое собственное лексическое окружение
}
alert(i); //Ошибка

// Блоки кода
{
  let message = "Hello";
  alert(message); //hello
}
alert(message)(
  //ошибка

  // IIFE immediately invoked function expression

  function () {
    let message = "Hello";
    alert(message);
  }
)(); // вызовется сразу
// TG;
let result = 2009;
(function (value) {
  delete value;
  if (value) {
    result = value;
  }
})(262); // 262, при use strict будет ругаться на delete value – удаление value из глобального объекта
```

# Construction Function

при создании однотипных объектов можно воспользоваться функцией конструктором через "new"

имя функции-конструктора начинается с большой буквы 2. должна вызываться через оператор new
можно вызывать без скобок, если нет аргументов

```js
function User(name) {
  this.name = name;
  this.isAdmin = false;
}
let user = new User("Вася");
alert(user.name); // Вася  alert(user.isAdmin); // false  происходит следующее:
// Создаётся новый пустой объект, и он присваивается this.
// Выполняется код функции. Обычно он модифицирует this, добавляет туда новые свойства.
// Возвращается значение this. 4. При вызове return с объектом, будет возвращён объект, а не this

function User(name) {
  //this = {}; (неявно)
  //добавляет к this  this.name = name;  this.isAdmin = false;
  //return this(неявно)
}
// Любая функция может быть использована как конструктор

new (function () {})();
let user = new (function () {
  this.name = "John";
  this.isAdmin = false;
  //	и т.д. такой конструктор вызывается один раз
})();
```

Используя специальное свойство new.target внутри функции мы можем проверить вызвана ли функция при помощи опреатора new или без него, если да, то в new.target будет сама функиця, в противном случае undefined

```js
function User() {
  alert(new.target);
}
User(); //undefined
new User(); // код User

// функцию можно вызывать как с new так и без него
function User(name) {
  if (!new.target) {
    return new User(name);
  }

  this.name = "name";
}

let john = User("John");
alert(john.name); // John

// Без new можно войти в заблуждение по поводу создания объекта
```

задача конструкторов – записать все необходимое в this
при вызове return с объектом будет возвращен объект а не this при вызове return с примитивным значением, оно будет отброшено

```js
function BigUser() {
  this.name = "Вася";
  return { name: "Godzilla" }; // <— возвращает этот объект
}

alert(new BigUser().name); // Godzilla, получили этот объект

function SmallUser() {
  this.name = "Вася";
  return; // <— возвращает this
}
alert(new SmallUser().name); // Вася
// При вызове return с примитивности значением , примитивное значение - отбросится
```

## Методы в конструкторе

```js
function User(name) {
  this.name = name;
  this.sayHi = function () {
    alert("Меня зовут: " + this.name);
  };
}
let vasya = new User("Вася");
vasya.sayHi();
// Меня зовут: Вася /* vasya = { name: "Вася", sayHi: function() { ... } }*/
```

# Eval

```js
// Синтаксис
let result = eval("code");
// Коду через eval доступны внешние переменные
let a = 1;
function f() {
  let a = 2;
  eval("alert(a)");
}
f();
let x = 5;
eval("x=10");
alert(x); //10
// Внутри eval переменные не видны, так как у eval свое лексическое окружение Если код внутри eval не использует внешние переменные, то лучше вызывать как

window.eval;

let x = 1;

{
  let x = 5;
  window.eval("alert(x)"); //1
}
// если нужны локальные переменные, то лучше использовать new Functions
```

# Function Expression и Function Declaration

```js
function sayHi() {
  alert("hi");
}
let func = sayHi; // если бы было let func = sayHi(), то мы бы присвоили func значение вызова sayHi()
func(); //hi
sayHi(); //hi
```

- FD – Объявление функции через ключевое слово function foo(){…}, FE – создание функции через присваивание некой переменной let foo = function(){…}.
- FE создает тогда, когда выполнение доходит до нее, в отличие от FD, которая всплывает, то есть мы не можем обратиться к функции, объявленной с помощью FD раньше ее создания.
- Видимость FD ограничивается {}, этот нюанс можно обойти с помощью присвоения переменной объявленной вне блока когда, а внутри блока с помощью FE присвоить функцию

```js
// При FE в конце кода должна стоять ;
let sayHi = function () {
  alert("Привет");
};
let func = sayHi;
```

# Function object

```js
// Свойство name
function sayHi() {
  alert("hi");
}
alert(sayHi.name); //SayHi;

let sayHi = function () {
  alert("Hi");
};
alert(sayHi.name);
```

```js
// Свойство length

// Встроенное свойство length содержит количество параметров функции
function f1(a) {}
function f2(a, b) {}
function many(a, b, ...more) {}
alert(f1.length); //1  alert(f2.length);//2
alert(many.length); //2
```

```js
function makeCounter() {
  let count = 0;

  function increaser() {
    return count++;
  }

  increaser.set = (value) => (count = value); //упаковка методов в increaser в силу того, что они разделяют  одну область видимости переменных
  increaser.decrease = () => count--;

  return increaser;
}
let counter = makeCounter();
alert(counter());
alert(counter());

counter.set(10); // работает так как counter возвращает функцию increaser  counter.decrease();

alert(counter());
```

```js
// Добавление пользовательских свойств с безымянными функциями и именными с возвратом
// - у них нет this

function wrap1() {
  return function () {
    wrap.userProp = "x"; // не получится добавить пользовательские свойства
  };

  function wrap1() {
    wrap.userProp = obj;

    function wrap2(...arg) {
      // подкинем args  wrap2.userProp = …
    }

    return wrap2;
  }
}
```

# new Function

Синтаксис
let func = new Function([arg1, arg2, …argN], functionBody);

```js
new Function("a", "b" "return a + b");
new Function("a, b" "return a + b");
new Function("a, b", "return a + b");

let sum = new Function("a", "b", "return a+b");  alert ( sum(1,2));
```

Главное отличие, что функция создается из строки, на лету

Замыкание
функция запоминает, где родилась в свойстве [[Environment]] Это ссылка на внешнее лексическое окружение new function имеет доступ только к глобальным переменным

```js
function getFunc() {
  let value = test;
  let func = new Function("alert(value)");
  let func = function () {
    alert(value);
  };
  return func;
}

getFunc()(); // ошибка value не определенно | "test" из ЛО функции getFunc

// Если бы new Function имела доступ к внешним переменным, возникли бы проблемы с минификатором
```

# NFE

```js
let sayHi = function func(who) {
  alert(`Helдo,${who}`);
};

// это позволяет ссылаться функции саму на себя и оно не доступно за пределами функции

let sayHi = function func(who) {
  if (who) {
    alert(`hello ${who}`);
  } else {
    func("guest");
  }
}; //так как FE

func(); // так работать не будет func не доступна вне функции

// Преимущество заключается в том, что sayHi может быть изменено

let sayHi = function (who) {
  if (who) {
    alert("Hello ${who}");
  } else {
    sayHi("Guest");
  }
};
let welcome = sayHi;
sayHi = null;
welcome(); //Не работает
```

!!!Не работает с FD

# Optional chain. Опциональная цепочка '?'

это безопасный способ доступа к свойствам вложенных объектов

```js
let user = {};
alert(user.address.street); //ошибка
alert(user && user.address && user.address.street); //undefined теперь без ошибки  с помощью опциональной цепочки
let user = {};
alert(user?.address); //undefined
alert(user?.address.street); //undefined
```

синтаксис ?. делает необязательным только свойство перед ним

```js
// ?.() – для работы с потенциально несуществующей функцией
let user1 = {
  admin() {
    alert("admin");
  },
};
let user2 = {};
user1.admin?.(); //admin
user2.admin?.(); //ничего, но вычисление продолжится без ошибок

// ?.[]
let user1 = { firstName: "John" };
let user2 = null;
let key = "firstName";
alert(user?.[key]); //John
alert(user?.[key]); // undefined
alert(user1?.[key]?.something?.not?.existing); //undefined
```

# Параметры функции

```js
function showMenu(title = "Untitled", width = 200, height = 100, items = []) {}
// плохой пример так как вызов будет выглядеть вот так, при вызове функции  showMenu("My Menu", undefined, undefined, [item1, item2])

// Можно передать параметры как объект
let options = {
  title: "My Menu",
  items: ["Item1", "Item2"],
};

function showMenu({
  title = "Untitled",
  width = 200,
  height = 100,
  items = [],
}) {
  alert(`${title} ${width} ${height}`); //My menu 100 200
  alert(items);
}
showMenu(options);

// Более сложный пример со вложенными объектами
let options = {
  title: "My Menu",
  items: ["Item1", "Item2"],
};

function showMenu({
  title = "Untitled",
  width: w = 100,
  height: h = 200,
  items: [item1, item2],
}) {
  alert(`${title} ${w} ${h}`);
}

shoeMenu(options);

// Полный синтаксис – такой же как и лля деструктурирующего объекта
function func({ incomingProperty: varName = defaultName}){}

// Если нам нужны все значения по умолчанию, то нужно передать пустой объект
showMenu({}) //Ок
showMenu() //Ошибка  или
function showMenu({title = "Menu", width = 100. height = 200} = {}){
}

showMenu();
```

```js
// TG;
const myFunc = ({ x, y, z }) => {
  console.log(x, y, z);
};
myFunc(1, 2, 3); //undefined, undefined, undefined, так как ждет объект, а получает три числа

TG;
const value = { number: 10 };
const multiply = (x = { ...value }) => {
  console.log((x.number *= 2));
};
multiply(); //20 так как не передан аргумент, value находим выше
multiply(); //20 аналогично, не меняя свойство number
multiply(value); //20 здесь изменяется объект
multiply(value); //40
```

# Rest params

Остаточные параметры передаются в массив :
function func(par1, par2,..., parN....arg)
Те которые не используются передаются в массив arg function func(...arg)
все переданные параметры в функцию перейдут в массив arg

Переменная arguments
Все аргументы функции находятся в псевдо массиве(не можем работать как с массивом) arguments под своими порядковыми номерами:

```js
function showName() {
  alert( arguments.length );
  alert( arguments[0] );
  alert( arguments[1] );}
function f(arg1, ...rest, arg2) { // arg2 после ...rest ?! // Ошибка
}

```

!!! rest должен быть последним
!!! стрелочные функции не имеют arguments, как и не имеют собственного this

```js
// Оператор расширения
let arr = [1,2,3];
Math.max(arr);// так не сработает
Math.max(…arr); //так сработает  let arr1 = [1, -2, 3, 4];
let arr2 = [8, 3, -8, 1];
alert( Math.max(...arr1, ...arr2) ); // 8
let merged = [0, ...arr, 2, ...arr2];  alert(merged);

let str = "Привет";
alert( [...str] ); // П,р,и,в,е,т


```

```js
const func = function () {
  return arguments;
};
console.log(func()); //object arguments
TG;
function foo() {
  arguments.forEach((el) => {
    console.log(el);
  });
}
foo(2, 3, 6); //Uncaught TypeError: arguments.forEach is not a function т.к. псевдомассив
```

Можно также это сделать с помощью Array.from(obj) – он работает и с псевдо массивами и итер. объектами

# Function BP. рекурсивный обход

```js
let company = {
  sales: [
    { name: "John", salary: 1000 },
    { name: "Alice", salary: 600 },
  ],
  development: {
    sites: [
      { name: "Peter", salary: 2000 },
      { name: "Alice", salary: 1300 },
    ],
  },
};

function sumSalaries(department) {
  if (Array.isArray(department)) {
    //Если объект - массив
    return department.reduce((prev, current) => prev + current.salary, 0);
    //Если это массив, то сложить все salary
  } else {
    let sum = 0;
    for (let subdep of Object.values(department)) {
      //для каждого суб-отдела повторить, проверить массив ли это
      sum += sumSalaries(subdep); // рекурсия
    }
    return sum;
  }
}

alert(sunSalaries(company));
```

# Functional properties

```js
// Пользовательские свойства
function sayHi() {
  alert("Hi");
  sayHi.counter++;
}
sayHi.counter = 0;
sayHi();
sayHi();
alert(`вызвана ${sayHi.counter} раза`); //функция счетчик, через свойство
function makeCounter() {
  function counter() {
    return counter.count++;
  }
  counter.count = 0;
  return counter;
}
let counter = makeCounter();
counter.count = 10; //в этом и заключается преимущество над обычной переменной. Мы можем  получить свойство вне блока кода
console.log(counter.count); //10
```

Если переменную во внутреннем лексическом окружении мы не можем изменить извне, то это
можно сделать с помощью вложенных функций или свойств

```js
// TG Передача по ссылке
class Counter {
  constructor() {
    this.count = 0;
  }
  increment() {
    this.count++;
  }
}
const counterOne = new Counter();
counterOne.increment();
counterOne.increment();
const counterTwo = counterOne; //создается ссылка на тот же самый объект
counterTwo.increment(); //увеличиваем еще на один
console.log(counterOne.count); //3
```
