# Worker()

```js
var myWorker = new Worker(aURL, {
  type: "classic",
  credentials: "omit",
  _name: "",
});
```

# Методы

## postMessage()

```js
worker.postMessage(
  message, //пересылаемы данные
  [transfer]
);
```

## terminate()

Завершает работу

```js
var myWorker = new ChromeWorker(self.path + "myWorker.js");

function handleMessageFromWorker(msg) {
  console.log("входящее сообщение от работника:", msg);
  switch (msg.data.aTopic) {
    case "do_sendMainArrBuff":
      sendMainArrBuff(msg.data.aBuf);
      break;
    default:
      throw "свойство aTopic отсутствует в сообщении ChromeWorker";
  }
}

myWorker.addEventListener("message", handleMessageFromWorker);

// Создание и отправка буфера
var arrBuf = new ArrayBuffer(8);
console.info("arrBuf.byteLength, ДО передачи:", arrBuf.byteLength);

myWorker.postMessage(
  {
    aTopic: "do_sendWorkerArrBuff",
    aBuf: arrBuf, // буфер который передаётся 3 строками ниже
  },
  [
    arrBuf, // буфер созданный на строке 9
  ]
);

console.info("arrBuf.byteLength, ПОСЛЕ передачи:", arrBuf.byteLength);
```

```js
// worker.js
self.onmessage = function (msg) {
  switch (msg.data.aTopic) {
    case "do_sendWorkerArrBuff":
      sendWorkerArrBuff(msg.data.aBuf);
      break;
    default:
      throw "свойство aTopic отсутствует в сообщении ChromeWorker";
  }
};

function sendWorkerArrBuff(aBuf) {
  console.info(
    "от рабочего, ДО отправки обратно, aBuf.byteLength:",
    aBuf.byteLength
  );

  self.postMessage({ aTopic: "do_sendMainArrBuff", aBuf: aBuf }, [aBuf]);

  console.info(
    "от рабочего, ПОСЛЕ отправки обратно, aBuf.byteLength:",
    aBuf.byteLength
  );
}
```

# События

## error

```js
addEventListener("error", (event) => {});

onerror = (event) => {};
```

## message

```js
var myWorker = new Worker("worker.js");

first.onchange = function () {
  myWorker.postMessage([first.value, second.value]);
  console.log("Сообщение, отправленное в worker-объект");
};

myWorker.onmessage = function (e) {
  result.textContent = e.data;
  console.log("Сообщение полученное из worker-объекта");
};
```

## messageerror

```js
addEventListener("messageerror", (event) => {});

messageerror = (event) => {};
```

# BPs.

## PB. пример

Позволяет запустить задачу в фоновом режиме

```html
<!-- форма из двух полей -->
<form>
  <div>
    <label for="number1">Multiply number 1: </label>
    <input type="text" id="number1" value="0" />
  </div>
  <div>
    <label for="number2">Multiply number 2: </label>
    <input type="text" id="number2" value="0" />
  </div>
</form>
```

```js
// main.js
// получаем значения
const first = document.querySelector("#number1");
const second = document.querySelector("#number2");

const result = document.querySelector(".result");

if (window.Worker) {
  const myWorker = new Worker("worker.js");

  first.onchange = function () {
    // отправляем в worker на обработку
    myWorker.postMessage([first.value, second.value]);
    console.log("Message posted to worker");
  };

  second.onchange = function () {
    // отправляем в worker на обработку
    myWorker.postMessage([first.value, second.value]);
    console.log("Message posted to worker");
  };

  // получаем
  myWorker.onmessage = function (e) {
    result.textContent = e.data;
    console.log("Message received from worker");
  };
} else {
  console.log("Your browser doesn't support web workers.");
}
// worker.js
onmessage = function (e) {
  console.log("Worker: Message received from main script");
  const result = e.data[0] * e.data[1];
  if (isNaN(result)) {
    postMessage("Please write two numbers");
  } else {
    const workerResult = "Result: " + result;
    console.log("Worker: Posting message back to main script");
    postMessage(workerResult);
  }
};
```

```ts
type WorkerParams = {
  URL: string; //адрес файла
  options: {
    type: "classic" | "module";
    credentials: "omit" | "same-origin" | "include";
    _name: string;
  };
};
// методы
const worker = new Worker(params: WorkerParams);
// отправит данные worker
worker.postMessage(message: {
  data: any
}, transfer: [ArrayBuffer,MessagePort, ImageBitmap ])
// завершит работу
worker.terminate()

worker.addEventListener("error", (event) => {});
worker.addEventListener("message", (event) => {});
worker.addEventListener("messageerror", (event) => {}); //для неопределенных ошибок

```

## Service Workers

Позволяет управлять приложением, может конфигурировать запросы. Жизненный цикл:

- Загрузка
- Установка
- Активация

Требования: HTTPS или localhost

## Web Workers

Позволяют производить вычисления в фоновом режиме не затрагивая интерфейс
