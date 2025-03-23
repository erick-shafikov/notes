# Promise

- У объекта promise, возвращаемого конструктором есть внутренние свойства:
- - state(состояние) - в начале pending (ожидание), которое меняется либо fulfilled или reject
- - result - в начале undefined, а потом меняется на value либо на error
- объект функции с двум я аргументами , которые принимают два значения state и result
- resolve и reject ожидает только один аргумент
- - resolve(value) - коллбек, вызванный, если работа завершилась успешно c результатом value
- - reject(error) - если произошла ошибка лучше вызывать с объектом Error
- может быть только что-то одно, исполнится только первый
- Promise.length === 1 всегда

Исполнитель запускается автоматически, а затем он должен вызвать resolve или reject.

```js
let promise = new Promise(function (resolve, reject) {
  resolve(value);
  reject(error);
});

promise
  .then(
    function (result) {},
    function (error) {}
  ) //работает с Thenable объектами
  .catch(f) //тоже самое, что и .then(null, errorHandlingFunction)
  .finally(f); //тоже самое, что и then(f,f)
```

Пример с setTimeout и успешным выполнением:

```js
let Promise = new Promise(function (resolve, reject) {
  //функция, которая вызывается автоматически
  setTimeout(() => resolve("done"), 1000);
});

// Пример с setTimeout и ошибкой:
let Promise = new Promise(function (resolve, reject) {
  setTimeout(() => reject(new Error("Упс")), 1000);
});
```

# Функции потребители

- then сам возвращает промис

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

# catch

если мы хотим обработать только ошибку, то есть метод .catch, который делает тоже самое, что и .then(null, errorHandlingFunction)

```js
let promise = new Promise(function (resolve, reject) {
  setTimeout(() => reject(new Error("Упс"), 1000));
}); //.catch(f) == promise.then(null, f)
promise.catch(alert);
```

Ошибки выброшенные из асинхронных функций не будут пойманы методом catch

```js
var p2 = new Promise(function (resolve, reject) {
  setTimeout(function () {
    throw "Uncaught Exception!";
  }, 1000);
});

p2.catch(function (e) {
  console.log(e); // Никогда не вызовется
});
```

Ошибки выброшенные после выполнения промиса будут проигнорированны

```js
var p3 = new Promise(function (resolve, reject) {
  resolve();
  throw "Silenced Exception!";
});

p3.catch(function (e) {
  console.log(e); // Никогда не вызовется
});
```

# finally

- finally(f) схож с .then(f, f), но не тоже самое
- finally не имеет аргументов
- finally пропускает ошибки и результат дальше

```js
new Promise((resolve, reject) => {
  setTimeout(() => resolve("done"), 2000);
})
  .finally(() => console.log("Промис завершен"))
  .then((result) => console.log(result));

new Promise((resolve, reject) => {
  throw new Error("error");
})
  .finally(() => console.log("промис завершен"))
  .catch((err) => console.log(err));
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
new Promise(function (resolve, reject) {
  setTimeout(() => resolve(1), 1000); //setTimeout должен быть внутри промиса
})
  .then(function (result) {
    console.log(result); //1

    return new Promise((resolve, reject) => {
      setTimeout(() => resolve(result * 2), 1000);
    });
  })
  .then(function (result) {
    console.log(result); //2

    return new Promise((resolve, reject) => {
      setTimeout(() => resolve(result * 2), 1000);
    });
  })
  .then(function (result) {
    console.log(result); //4
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
fetch("/url/")
  .then(response.json)
  .then((user) => fetch("/url/${user.name}"))
  .then((response) => response.json)
  .then(
    (githubUser) =>
      new Promise((resolve, reject) => {
        let img = document.createElement("img");
        img.src = githubUser.avatar_url;
        img.className = "promise-avatar-example";
        document.body.append(img);

        setTimeout(() => {
          img.remove();
          resolve(githubUser);
        }, 3000);
      })
  )
  .catch((error) => alert(error));
// Любой из ошибочных промисов будет отклонен
```

## Неявный try catch

Вокруг функции промиса и обработчиков находится «невидимый try...catch

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

# Promise[Symbol.species]

значение конструктора - this, на котором был вызван

```js
// псевдо код
class Promise {
  static get [Symbol.species]() {
    return this;
  }
}
```
