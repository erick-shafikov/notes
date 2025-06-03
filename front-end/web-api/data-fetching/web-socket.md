Протокол передачи данных без разрыва HTTP запроса, их адреса отличаются от обычных ws и wss вместо http(s). При первом запросе отправляется GET запрос с заголовком Upgrade: WebSocket (и другими хост, домен, протокол) и уникальный хэш. Нельзя эмулировать WS с помощью fetch или XMlHttpRequest. Можно настроить разный формат обмена данных во втором параметре в виде массива

```js
const socket = new WebSocket("ws://localhost:8080");

// ----------------------------------------------------------------------
//конструктор

const socket = new WebSocket(url); // url
const socket = new WebSocket(url, protocols); // строка или массив строк с протоколами ["soap", "wamp"]

// ----------------------------------------------------------------------
// свойства экземпляра

socket.binaryType; //"blob" или "arraybuffer" тип данных в соединении

socket.addEventListener("message", (event) => {
  if (event.data instanceof ArrayBuffer) {
    // binary frame
    const view = new DataView(event.data);
    console.log(view.getInt32(0));
  } else {
    // text frame
    console.log(event.data);
  }
});

socket.bufferedAmount; //количество байт в очереди
socket.extensions; // расширения выбранные сервером
socket.protocol; // Подпротокол
socket.readyState; // состояние соединения
// 0 – «CONNECTING»: соединение ещё не установлено,
// 1 – «OPEN»: обмен данными,
// 2 – «CLOSING»: соединение закрывается,
// 3 – «CLOSED»: соединение закрыто.

socket.url; //абсолютный url

// ----------------------------------------------------------------------
// методы

socket.close(code, reason); //закрыть соединение
// code от 1001 0 1015
// reason строка описание закрытия
socket.send(data); //добавить в очередь для отправки
// data может быть string, ArrayBuffer, Blob, DataView, TypedArray
```

События:

- close
- error
- message
- open

```ts
type WebSocketCloseEvent = {
  code: number;
  reason: string;
  wasClean: boolean;
};

type WebSocketMessage = {
  data: sting | Buffer | ArrayBuffer;
  origin: string;
  lastEventId: string;
  source: string;
  ports: string;
};

// Соединение открыто
socket.addEventListener("open", (event: Event) => {
  socket.send("Hello Server!");
});

// Получение сообщений
socket.addEventListener("message", (event: WebSocketMessage) => {
  console.log("Message from server ", event.data);
});

// закрытие канала
socket.addEventListener("close", (event: WebSocketCloseEvent) => {
  console.log("Message from server ", event.data);
});

// Получение сообщений
socket.addEventListener("error", (event: Event) => {
  console.log("Message from server ", event.data);
});
```
