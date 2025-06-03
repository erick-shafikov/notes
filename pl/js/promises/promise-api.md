<!-- Promise API ----------------------------------------------------------------------------------------------------------------------------->

# Promise.all

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
let names = ["John", "Garry", "Max"];
let requests = names.map((name) => fetch("url/"));
Promise.all(requests)
  .then((responses) => {
    for (let response of responses) {
      alert(`${response.url}:${response.status}`);
    }
    return responses;
  })
  .then((responses) => Promise.all(responses.map((r) => r.json())))
  .then((user) => users.forEach((user) => alert(user.name)));
```

- Если любой из промисов завершает с ошибкой, то промис возвращенный Promise.all немедленно завершиться с этой ошибкой

```js
Promise.all([
  new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
  new Promise((resolve, reject) =>
    setTimeout(() => reject(new Error("Ошибка")), 2000)
  ),
  new Promise((resolve, reject) => setTimeout(() => resolve(3), 3000)),
]).catch(alert); //Error: Ошибка, все остальные результаты игнорируются
```

- если один из объектов не является промисом, он передается в итоговый массив как есть

```js
Promise.all([new promise((resolve, reject) => resolve(1), 1000), 2, 3]).then(
  alert
); //1,2,3
```

# Promise.allSettled

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
  (results) => {
    results.forEach((result, num) => {
      if (result.status === "fulfilled") {
        alert(`${urls[num]}:${result.value.status}`);
      }

      if (result.status === "rejected") {
        alert(`${urls[num]}:${result.reason}`);
      }
    });
  }
);

// массив
// results:  [ {status:"fulfilled", value:…}, {status:"fulfilled", value:…}, {status:"rejected", value: ошибка} ]
```

# Promise.race

ждет только первый промис из которого берет результат или ошибку

```js
let promise = Promise.race(iterable);

Promise.race([
  new Promise((resolve, reject) => setTimeout(() => resolve(1), 1000)),
  new Promise((resolve, reject) =>
    setTimeout(() => reject(new Error("Ошибка")), 2000)
  ),
  new Promise((resolve, reject) => setTimeout(() => resolve(2), 3000)),
]).then(alert); //1
```

быстрее всех выполнился первый промис, а на втором возникла ошибка

# Promise.any

Принимает итерируемый объект, как только один из промисов разрешится удачно, то вернет его. В отличие от Promise.race вернет первый успешный

при всех неудачных - AggregateError и Array с причинами

# Promise.resolve

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
```

```js
Promise.resolve("Success").then(
  function (value) {
    console.log(value); // "Success"
  },
  function (value) {
    // не будет вызвана
  }
);
//с массивом
var p = Promise.resolve([1, 2, 3]);
p.then(function (v) {
  console.log(v[0]); // 1
});
//Выполнение с другим промисом ( Promise)
var original = Promise.resolve(true);
var cast = Promise.resolve(original);
cast.then(function (v) {
  console.log(v); // true
});
```

# Promise.reject()

```js
Promise.reject(new Error("провал")).then(
  function (success) {
    // не вызывается
  },
  function (error) {
    console.log(error); // печатает "провал" + Stacktrace
    throw error; // повторно выбрасываем ошибку, вызывая новый reject
  }
);

Promise.reject(error); //создает промис завершенный с ошибкой error тоже самое, что и
let promise = new Promise((resolve, reject) => reject(error));
```

# Promise.try()

обмачивает в промис

```js
Promise.try(func);
Promise.try(func, arg1);
Promise.try(func, arg1, arg2);
Promise.try(func, arg1, arg2, /* …, */ argN);
```

# Promise.withResolvers()

альтернатив для

```js
let resolve, reject;
const promise = new Promise((res, rej) => {
  resolve = res;
  reject = rej;
});
```

```js
async function* readableToAsyncIterable(stream) {
  let { promise, resolve, reject } = Promise.withResolvers();
  stream.on("error", (error) => reject(error));
  stream.on("end", () => resolve());
  stream.on("readable", () => resolve());

  while (stream.readable) {
    await promise;
    let chunk;
    while ((chunk = stream.read())) {
      yield chunk;
    }
    ({ promise, resolve, reject } = Promise.withResolvers());
  }
}
```
