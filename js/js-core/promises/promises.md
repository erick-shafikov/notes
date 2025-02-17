```js
let promise = new Promise(function (resolve, reject) {
  //возвращает объект с двумя  свойствами status и value
  resolve(value);
  reject(error);
}); //функции cb, которые принимают два значения state и result
//может быть только что-то одно, исполнится только первый

//Недоступная часть
// Promise.state: pending | fulfilled | reject
// Promise.result: undefined | fulfilled | error
//обработчики для случаев value и error
promise
  .then(
    function (result) {},
    function (error) {}
  ) //работает с Thenable объектами
  .catch(f) //тоже самое, что и .then(null, errorHandlingFunction)
  .finally(f); //тоже самое, что и then(f,f)
```

- resolve(value) - коллбек, вызванный, если работа завершилась успешно c результатом value
- reject(error) - если произошла ошибка

Исполнитель запускается автоматически, а затем он должен вызвать resolve или reject. У объекта promise, возвращаемого конструктором есть внутренние свойства

- state(состояние) - в начале pending(ожидание), которое меняется либо fulfilled или reject
- result - в начале undefined, а потом меняется на value либо на error

Пример с setTimeout и успешным выполнением:

```js
let Promise = new Promise(function(resolve, reject){
//функция, которая вызывается автоматически
setTimeout(()=>resolve("done"),1000;);
});


// Пример с setTimeout и ошибкой:
let Promise = new Promise(function(resolve, reject){
  setTimeout(()=> reject (new Error("Упс")), 1000);
});

```

- !!!Может быть что-то одно, все остальное игнорируется
- !!!resolve и reject ожидает только один аргумент
- !!!reject лучше вызывать с объектом Error

# Функции потребители

```js
promise.then(
  function (result) {
    //обработает успешное выполнение
  },
  function (error) {
    //обработает ошибку
  }
);
```

```js
// Пример с удачным исполнения промиса
let promise = new Promise(function (resolve, reject) {
  setTimeout(() => resolve("done"), 1000);
});

promise.then(
  (result) => alert(result) //выведет done
  error => alert(error)
);
```

```js
// пример с неудачным
let promise = new Promise(function (resolve, reject) {
  setTimeout(() => reject(new Error("Упс!"), 1000));
});

promise.then(
  (result) => alert(result), //не будет запущена error
  (error) => alert(error) //Упс? спустя секунду
);
```

```js
// Если нам нужно обработать только результат, при успешном выполнение, то можно передать только один  аргумент в
let promise = new Promise(function(resolve = > {
  setTimeout(()=> resolve("done!"), 1000);
}));

promise.then(alert);
```

## catch

метод .catch
если мы хотим обработать только ошибку, то есть метод .catch, который делает тоже самое, что и .then(null, errorHandlingFunction)

```js
let promise = new Promise(function (resolve, reject) {
  setTimeout(() => reject(new Error("Упс"), 1000));
}); //.catch(f) == promise.then(null, f)  promise.catch(alert)
```

## finally

```js
finally(f) схож с .then(f, f), но не тоже самое
finally не имеет аргументов
finally пропускает ошибки и результат дальше

new Promise((resolve, reject)=>{  setTimeout(()=>resolve("done"), 2000)
})
.finally(()=>alert("Промис завершен"))
.then(result=>alert(result));

new Promise((resolve, reject)=>{  throw new Error("error");
})
.finally(()=>alert("промис завершен"))
.catch(err => alert(err));

```

# Promises chain

Код выполняется, так как promise.then возвращает промис, когда обработчик возвращает какое-то значение, то оно становится результатом выполнения

```js
new Promise(function (resolve, reject) {
  setTimeout(() => resolve(1), 1000); //первый промис выполняется через 1 секунду
})
  .then(function (result) {
    //вызывается обработчик
    alert(result); //1
    return result * 2;
  })
  .then(function (result) {
    //возвращаемый результата в следующий обработчик
    alert(result); //2
    return result * 2;
  })
  .then(function (result) {
    alert(result); //4
    return result * 2;
  });
```

## Возвращаемые промисы

```js
new Promise(function(resolve, reject){
  setTimeout(() => resolve(1), 1000); //setTimeout должен быть внутри промиса
}).then(function(result)){
  alert(result); //1

  return new Promise((resolve, reject) => {
setTimeout(()=> resolve(result * 2), 1000);
})}


.then(function(result){
  alert(result); //2

return new Promise((resolve, reject) => {
  setTimeout(()=>resolve(result * 2), 1000);
});

}).then(function(result){

alert(result); //4

});

```

# thenable

Обработчик может возвращать не именно промис, а любой объект, содержащий метод then. Этот объект будет
обработан как промис.

```js
class Thenable {
  constructor(num) {
    this.num = num;
  }
  then(resolve, reject) {
    alert(resolve);
    setTimeout(() => resolve(this.num * 2), 1000);
  }
}

new Promise((resolve) => resolve(1))
  .then((result) => {
    return new Thenable(result); //JS проверяет объект возвращаемый из обработчика then
  })
  .then(alert); //покажет 2 через 200мс
```

# Обработка ошибок

Если промис завершился с ошибкой, то управление переходит в ближайший обработчик ошибок, чтобы перехватить ошибку, catch можно переместить в конец:

```js
fetch(/url/)
.then(response.json)
.then(user=>fetch("/url/${user.name}"))
.then(response => response.json())
.then(githubUser => new Promise(resolve, reject)=>{
  let img = document.createElement("img");
  img.src = githubUser.avatar_url;
  img.className = "promise-avatar-example";
  document.body.append(img);

  setTimeout(()=>{
    img.remove();
    resolve(githubUser);
}, 3000);
})

.catch(error=>alert(error));
// Любой из ошибочных промисов будет отклонен

```

## Неявный try catch

Вокруг функции промиса и обработчиков находится «невидимый try..catch»

```js
new Promise((resolve, reject) => {
  throw new Error("Ошибка!");
}).catch(alert);

// тоже самое, что и:

new Promise((resolve, reject) => {
  reject(new Error("Ошибка!"));
}).catch(alert);

// Если мы бросим ошибку из обработчика then, то промис будет отклоненным и обработка ошибок перейдет к  ближайшему обработчику ошибок:

new Promise((resolve, reject) => {
  resolve("ок");
})
  .then((result) => {
    throw new Error("Ошибка");
  })
  .catch(alert); //Error: Ошибка
```

## Проброс ошибок

.catch ведет себя как try catch, можно использовать сколько угодно .then и в конце один catch. Если мы пробросим throw ошибку внутри блока .catch, то управление перейдет к следующему ближайшему обработчику ошибок, а если мы обработаем ошибку, то продолжит работу ближайший обработчик .then

```js
new Promise((resolve, reject) => {
  throw new Error("Ошибка");
})
  .catch(function (error) {
    alert("Ошибка обработана продолжить работу");
    //ошибка обработана, блок .catch завершился нормально, поэтому вызывается следующий обработчик .then
  })
  .then(() => alert("Управление перейдет в следующий then"));

// Теперь .catch не может обработать ошибку и пробрасывает ее

new Promise((resolve, reject) => {
  throw new Error("Ошибка");
})
  .catch(function (error) {
    if (error instanceof URIError) {
      //обработка ошибки
    } else {
      alert("Не могу обработать ошибку!");

      throw error;
    }
  })
  .then(function () {
    // не выполнится
  })
  .catch((error) => {
    alert("Неизвестная ошибка"); //Ничего не возвращаем выполнение продолжается в нормальном режиме/
  });
```

## Необработанные ошибки

Если ошибку не обработать, то код падает, в браузере мы можем поймать такие ошибки с помощью unhandledrejectiion:

```js
window.addEventListener("unhandledrejectiion", function (event) {
  alert(event.promise); //[object Promise] объект, который сгенерировал ошибку
  alert(event.reason); //Error: Ошибка - объект ошибки, который не был обработан
});

new Promise(function () {
  throw new Error("Ошибка");
}); //нет обработчика ошибок
```

<!-- Promise API ----------------------------------------------------------------------------------------------------------------------------->

# Promise API:

## Promise.all

Принимает массив промисов (или любой перебираемый объект) и возвращает промис. Новый промис
завершиться, когда завершаться все промисы и его результатом будет их массив.

```js
Promise.all([
  new Promise((resolve) => setTimeout(() => resolve(1)), 3000),
  new Promise((resolve) => setTimeout(() => resolve(2)), 2000),
  new Promise((resolve) => setTimeout(() => resolve(3)), 1000),
]).then(alert);
```

- Когда все массивы выполнятся результатом будет массив [1,2,3]
- Порядок элементов в массиве соответствует порядку ДОБАВЛЕННЫХ промисов

пропуск массива через map-функцию

```js
let urls = ["html1", "html2", "html3"];
let requests = urls.map((url) => fetch(url)); //для каждого применить fetch, получить промис response
Promise.all(requests).then((responses) =>
  responses.forEach(
    // для каждого response вывести url и статус
    (responses) => alert("${responses.url}:${responses.status}")
  )
);
// Пример 2:
let names = ["illiakan", "remy", "jeresig"];
let requests = names.map((name) => fetch("url/${name}"));
Promise.all(requests)
  .then((responses) => {
    for (let response of responses) {
      alert("${response.url}:${response.status}");
    }
    return responses;
  })
  .then((responses) => Promise.all(responses.map((r) => r.json())))
  .then((user) => users.forEach((user) => alert(user.name)));
```

Если любой из промисов завершает с ошибкой, то промис возвращенный Promise.all немедленно завершиться с этой ошибкой

```js
Promise.all([
  new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
  new Promise((resolve, reject) =>
    setTimeout(() => reject(new Error("Ошибка")), 2000)
  ),
  new Promise((resolve, reject) => setTimeout(() => resolve(3), 3000)),
]).catch(alert); //Error: Ошибка, все остальные результаты игнорируются

// если один из объектов не является промисом, он передается в итоговый массив как есть:

Promise.all([new promise((resolve, reject) => resolve(1), 1000), 2, 3]).then(
  alert
); //1,2,3
```

## Promise.allSettled

Promise.all подходит, когда нужно чтобы каждый из промисов выполнился правильно, Promise.all ждет
завершение всех промисов, а результат будет выгладить как:

```js
const res = [
  { status: "fulfilled", value: "результат" }, //в случае успешного выполнения
  { status: "rejected", reason: "ошибка" }, //в случае ошибки
];
```

```js
// Пример, когда нам нужно загрузить информацию о всех пользователях

let urls = ["https://user1", "https://user2", "https://no-such-url"];

//map возвращает массив результатов, для каждого применяя fetch, fetch возвращает промисы в виде response
const result = Promise.allSettled(urls.map((url) => fetch(url))).then(
  (result) => {
    results.forEach((results, num) => {
      if (result.status === "fulfilled") {
        alert(`${urls[num]}:${result.value.status}`);
      }

      if (results.status === "rejected") {
        alert($`{urls[num]}:${result.reason}`);
      }
    });
  }
);

// массив
// results:  [ {status:"fulfilled", value:…}, {status:"fulfilled", value:…}, {status:"rejected", value: ошибка} ]
```

## Promise.race

ждет только первый промис из которог берет результат или ошибку

```js
let promise = Promise.race(iterable);

Promise.race([
  new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
  new Promise((resolve, reject) =>
    setTimeout(() => reject(new Error("Ошибка")), 2000)
  ),
  new Promise((resolve, reject) => setTimeout(() => reolve(2), 3000)),
]).then(alert); //1
```

быстрее всех выполнился первый промис, а на втором возникла ошибка

## Promise.resolve

Старая альтернатива async и await

Promise.resolve(value) создает успешно выполнившийся промис с результатом value, это тоже самое, что и  
let promise = new Promise(resolve => resolve(value));

LoadCached загружает url и кеширует его содержимое

```js
let cache = new Map();

function loadCached(url) {
  if (cache.has(url)) {
    return Promise.resolve(cache.get(url));
  }

  return fetch(url)
    .then((response) => response.text())
    .then((text) => {
      cache.set(url, text);
      return text;
    });
}

Promise.reject(error); //создает промис завершенный с ошибкой error тоже самое, что и let promise = new Promise((resolve, reject) => reject(error));
```

<!-- Промисификация -------------------------------------------------------------------------------------------------------------------------->

# Промисификация

Промисификация - изменение функции, которая принимает коллбек для возвращения промиса

```js
function loadScript(src, callback) {
  let script = document.createElement("script");
  script.src = src;

  script.onload = () => callback(null, script);
  script.onerror = () => callback(new Error("Ошибка загрузки скрипта ${src}"));

  document.head.append(script);
}

loadScript("/1.js/", (err, script) => {});

// Промисифицированный вариант:

let loadScriptPromise = function (src) {
  return new Promise((resolve, reject) => {
    loadScript(src, (err, script) => {
      if (err) reject(err); //если загрузка с ошибкой вернуть err
      else resolve(script); //если загрузка удачная, то вернуть script
    });
  });
};
```

<!-- async await ----------------------------------------------------------------------------------------------------------------------------->

# async await

```js
//Эта функция всегда возвращает промис, значения других типов оборачиваются в завершившийся промис автоматически
async function f() {
  return 1;
}
f().then(alert);

// тоже самое что и:
async function f() {
  return Promise.resolve(1);
}
f().then(alert); //1
```

await - заставляет ждать интерпретатор JS до тех пор, пока промис справа от await не выполнится, после чего он выполнит результат и исполнение кода продолжится :
let value = await promise;

!!! можно использовать только внутри функции async
!!! Нельзя использовать внутри обычных функций, но моно обернуть функцию

```js
(async () => {
  let response = await fetch("/url/");
})();

async function f() {
  let promise = new Promise((resolve, reject) => {
    setTimeout(() => resolve("Done"), 1000);
  });

  let result = await promise; //будет ждать, пока промис не выполнится

  alert(result);
}
f();
```

```js
// Пример showAvatar()

async function showAvatar() {
  let response = await fetch("/url/");
  let user = await response.json();

  let gitHubResponse = await fetch("/url/${user.name}");

  let gitHUbUser = await gutHubResponse.json();

  let img = document.createElement("img");
  img.src = gitHubUser.avatar_url;
  img.className = "promise-avatar-example";
  document.body.append(img);

  await new Promise((resolve, reject) => setTimeout(resolve, 3000));
  img.remove(); //следующая строка кода выполнится через 3 сек  return gitHubUser;
}

showAvatar();
```

## работа с thenable объектами

await работает с thenable объектами. Если у объекта можно вызвать Then, то этого достаточно, чтобы
использовать его c await

```js
class Thenable {
  constructor(num) {
    this.num = num;
  }
  then(resolve, reject) {
    alert(resolve);
    setTimeout(() => resolve(this.num * 2), 1000);
  }
}

async function f() {
  let result = await new Thenable(1);
  alert(result);
}

f();

// Асинхронные методы классов
class Waiter {
  async wait() {
    //для объявления асинхронного метода достаточно записать async перед именем
    return await Promise.resolve(1); //такой метод гарантированно возвращает промис, модно использовать await
  }
}
new Waiter().wait().then(alert); //1
```

## Обработка ошибок

```js
async function f() {
  await Promise.reject(new Error("Упс!"));
}
// тоже самое что и
async function f() {
  throw new Error("Упс!");
}
// промис может завершиться с ошибкой не сразу, ловить ошибки можно с помощью try

async function f() {
  try {
    let response = await fetch("/url/");
  } catch (err) {
    alert(err);
  }
}
f();

// несколько строк с await
async function f() {
  try {
    let response = await fetch("/url/");
    let user = await response.json();
  } catch (err) {
    alert(err);
  }
}

f();
// без try..catch
async function f() {
  let response = await fetch("/url/");
}
f().catch(alert);
```
