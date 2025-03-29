# аргументы функции

!!!Параметры по умолчанию undefined
!!!В параметрах может быть функция

## arguments

псевдо-массив

- можно присвоить значение arguments[n] = some_value

```js
function func() {
  arguments[0]; //обращение к аргументу
  arguments.length; //количество аргументов
}
```

```js
function myConcat(separator) {
  var result = "";
  var i;

  // iterate through arguments
  for (i = 1; i < arguments.length; i++) {
    result += arguments[i] + separator;
  }
  return result;
}

// возвращает "red, orange, blue, "
myConcat(", ", "red", "orange", "blue");

// возвращает "elephant; giraffe; lion; cheetah; "
myConcat("; ", "elephant", "giraffe", "lion", "cheetah");

// возвращает "sage. basil. oregano. pepper. parsley. "
myConcat(". ", "sage", "basil", "oregano", "pepper", "parsley");
```

```js
const func = function () {
  return arguments;
};
console.log(func()); //object arguments

function foo() {
  arguments.forEach((el) => {
    console.log(el);
  });
}
foo(2, 3, 6); //Uncaught TypeError: arguments.forEach is not a function т.к. псевдо-массив
```

!!! стрелочные функции не имеют arguments, как и не имеют собственного this

Можно также это сделать с помощью Array.from(obj) – он работает и с псевдо массивами и итерируемыми объектами

## аргументы по умолчанию

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

// Если нам нужны все значения по умолчанию, то нужно передать пустой объект
showMenu({}) //Ок
showMenu() //Ошибка  или
function showMenu({title = "Menu", width = 100. height = 200} = {}){
}

showMenu();
```

## в аргументах доступны другие аргументы

```js
function greet(name, greeting, message = greeting + " " + name) {
  return [name, greeting, message];
}

greet("David", "Hi"); // ["David", "Hi", "Hi David"]
greet("David", "Hi", "Happy Birthday!"); // ["David", "Hi", "Happy Birthday!"]
```

## объекты в качестве аргумента

```js
function myFunc(theObject) {
  theObject.make = "Toyota";
}

var mycar = { make: "Honda", model: "Accord", year: 1998 };
var x, y;

x = mycar.make; // x получает значение "Honda"

myFunc(mycar);
y = mycar.make; // y получает значение "Toyota", свойство было изменено функцией
```

```js
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
все переданные параметры в функцию перейдут в массив arg, rest -массив, arguments - псевдо массив

```js
function f(arg1, ...rest, arg2) {
  // arg2 после ...rest ?!
  // Ошибка
}
```

!!! rest должен быть последним

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

<!-- BPs: ------------------------------------------------------------------------------------------------------------------------------------>

# Optional chain ?

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
