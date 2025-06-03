broadcast api предоставляет возможность делится данными в одном источнике
позволяет настроить обмен сообщениями с iframe

```js
// создание и подключение к каналу
const bc = new BroadcastChannel("test_channel");
//отправка сообщения
bc.postMessage("This is a test message.");

// обработчик
bc.onmessage = (event) => {
  console.log(event);
};
//отключение
bc.close();
```

конструктора

BroadcastChannel(channelName)

```js
const bc = new BroadcastChannel("internal_notification");
bc.postMessage("New listening connected!");
```

свойства:

- name - имя

обработчики событий

- onmessage - событие срабатывающие на получение сообщения
- onmessageerror - событие срабатывающие на ошибку сообщения
- объект события:
- - data
- - origin - строка
- - lastEventId
- - source
- - ports

методы

- postMessage() - отправление любого типа объекта
- close() - закрыть канал
