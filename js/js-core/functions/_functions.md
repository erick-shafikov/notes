# Function Expression и Function Declaration

```js
function sayHi() {
  alert("hi");
}
let func = sayHi; // если бы было let func = sayHi(), то мы бы присвоили func значение вызова sayHi()
func(); //hi
sayHi(); //hi
```

- FD – Объявление функции через ключевое слово function foo(){}, FE – создание функции через присваивание некой переменной let foo = function(){}.
- FE создает тогда, когда выполнение доходит до нее, в отличие от FD, которая всплывает, то есть мы не можем обратиться к функции, объявленной с помощью FD раньше ее создания.
- Видимость FD ограничивается {}, этот нюанс можно обойти с помощью присвоения переменной объявленной вне блока когда, а внутри блока с помощью FE присвоить функцию
- Function declaration, если находятся внутри блоков {}, вне их не видны, обойти можно про использовании FE

```js
// При FE в конце кода должна стоять ;
let sayHi = function () {
  alert("Привет");
};
let func = sayHi;
```

# параметры функции

!!!Параметры по умолчанию undefined
!!!В параметрах может быть функция

```js
function showMessage(from, text = "текст не добавлен") {
  alert(from + ": " + text);
}
showMessage("Аня"); // Аня: текст не добавлен
```

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

// TG;
const value = { number: 10 };
const multiply = (x = { ...value }) => {
  console.log((x.number *= 2));
};
multiply(); //20 так как не передан аргумент, value находим выше
multiply(); //20 аналогично, не меняя свойство number
multiply(value); //20 здесь изменяется объект
multiply(value); //40
```

## rest

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
let arr = [1, 2, 3];
Math.max(arr); // так не сработает
Math.max(...arr); //так сработает  let arr1 = [1, -2, 3, 4];
let arr2 = [8, 3, -8, 1];
alert(Math.max(...arr1, ...arr2)); // 8
let merged = [0, ...arr, 2, ...arr2];
alert(merged);

let str = "Привет";
alert([...str]); // П,р,и,в,е,т
```

```js
const func = function () {
  return arguments;
};
console.log(func()); //object arguments
// TG;
function foo() {
  arguments.forEach((el) => {
    console.log(el);
  });
}
foo(2, 3, 6); //Uncaught TypeError: arguments.forEach is not a function т.к. псевдомассив
```

Можно также это сделать с помощью Array.from(obj) – он работает и с псевдо массивами и итер. объектами

# Closures

!!! Переменные объявленные внутри функции видны только внутри функции
!!! Может изменять значение внешних переменных, если внутри нет такой же переменной, меняет их после вызова функции, если внутри есть такая переменная, то не изменяет внешнюю.

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

# Function object

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
counter.count = 10; //в этом и заключается преимущество над обычной переменной. Мы можем получить свойство вне блока кода
console.log(counter.count); //10
```

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
      // подкинем args  wrap2.userProp =
    }

    return wrap2;
  }
}
```

<!-- new Function----------------------------------------------------------------------------------------------------------------------------->

# new Function

Синтаксис
let func = new Function([arg1, arg2, …argN], functionBody);

```js
new Function("a", "b", "return a + b");
new Function("a, b", "return a + b");
new Function("a, b", "return a + b");

let sum = new Function("a", "b", "return a+b");
alert(sum(1, 2));
```

Главное отличие, что функция создается из строки, на лету

Замыкание
функция запоминает, где родилась в свойстве Environment Это ссылка на внешнее лексическое окружение new function имеет доступ только к глобальным переменным

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

<!-- NFE ------------------------------------------------------------------------------------------------------------------------------------->

# NFE

!!!Не работает с FD

```js
let sayHi = function func(who) {
  alert(`Hello,${who}`);
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

<!-- BPs: ------------------------------------------------------------------------------------------------------------------------------------>

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

<!-- BPs: ------------------------------------------------------------------------------------------------------------------------------------>

# BPs:

## Add-hoc

Когда пользователь отвечает на вопрос – функция называется обработчиком. Можем передать 2 типа обработчиков – функцию без аргументов для положительного ответа и функцию с аргументами, которая будет вызываться в любом случае

```js
function ask(question, ...handlers) {
  // вопросы и обработчик
  let isYes = confirm(question); // isYes принимает либо true либо false

  for (let handler of handlers) {
    //для каждого из handlers
    if (handler.length == 0) {
      //если количество обработчиков = 0, то есть их нет, в этом примере не  выполнится эта часть
      if (isYes) handler(); //и если ответ true для isYes то вызывается, если false, то нет
    } else {
      // если handler не пустой, то передать true, как аргумент в каждый объект, 2 раза  выполнится эта часть, так как
      handler(isYes);
    }
  }
}

ask(
  "Вопрос?",
  () => alert("Вы ответили да"),
  (result) => alert(result)
); // в любом случае вызывается  вторая функция.
```

## BP. задачи

### рекурсивный обход

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
