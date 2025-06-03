# кеширование функции

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

# delay

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

# defer

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
