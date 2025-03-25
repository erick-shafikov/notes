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
f().then(alert);
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
```

## Асинхронные методы классов

```js
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

<!-- BPs --------------------------------------------------------------------------------------------------------------------------->

# BPs

## BPs mdn-пример

```js
function resolveAfter2Seconds(x) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(x);
    }, 2000);
  });
}

async function add1(x) {
  const a = await resolveAfter2Seconds(20);
  const b = await resolveAfter2Seconds(30);
  return x + a + b;
}

add1(10).then((v) => {
  console.log(v); // prints 60 after 4 seconds.
});

async function add2(x) {
  const a = resolveAfter2Seconds(20);
  const b = resolveAfter2Seconds(30);
  return x + (await a) + (await b);
}

add2(10).then((v) => {
  console.log(v); // prints 60 after 2 seconds.
});
```
