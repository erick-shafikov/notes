# WebRTC

технология позволяет браузерам обмениваться аудио и видео, состоит из:

- методы захвата аудио и видео
- методы передачи потоков. Поток - последовательность данных

# запрос устройств

```js
const videoEl = document.getElementById("cam-video-stream");

navigator.mediaDevices
  .getUserMedia({ audio: true, video: true })
  .then((stream) => {
    const constraints = { video: true };
    const stream = navigator.mediaDevices
      .getUserMedia(constraints)
      .then((stream) => {
        videoEl.srcObject = stream;
      });
  })
  .catch((error) => {
    console.error("Ошибка при получении медиа-потока:", error);
  });

// выбор устройства
navigator.mediaDevices.enumerateDevices().then((devices) => {
  console.log("Список медиа-устройств:", devices);
  /* Вывод
  Список медиа-устройств:
  [
    {
      "deviceId": "default",
      "kind": "audioinput",
      "label": "Default - Микрофон MacBook Pro (Built-in)",
      "groupId": "xxxxxxxxxxxxxxxxxxxxx"
    },
  ] */
});
```

```html
<video autoplay playsinline controls="false" id="cam-video-stream"></video>
```

# новые устройства

```js
let devices = [];

// Получаем список медиа-устройств при загрузке страницы
navigator.mediaDevices.enumerateDevices().then((newDevices) => {
  devices = newDevices;
  console.log("Список медиа-устройств:", devices);
});

// Устанавливаем обработчик события, который будет вызываться при изменении доступного списка устройств
navigator.mediaDevices.addEventListener("devicechange", () => {
  navigator.mediaDevices.enumerateDevices().then((newDevices) => {
    devices = newDevices;
    console.log("Обновленный список медиа-устройств:", devices);
  });
});
```

получение потоков

```js
const devices = await navigator.mediaDevices.enumerateDevices();
const streams = await navigator.mediaDevices
  .getUserMedia({
    audio: {
      deviceId: devices[0].deviceId, // Указываем конкретное устройство для аудио
    },
    video: {
      deviceId: devices[1].deviceId, // Указываем конкретное устройство для видео
    },
  })
  .catch((error) => {
    console.error("Ошибка при получении медиа-потока:", error);
  });
```

# настройки видео потока

```js
// настройки

const videoEl = document.getElementById("cam-video-stream");

const constraints = {
  audio: {
    deviceId: "default", // Используем устройство по умолчанию для аудио
    echoCancellation: true, // Включаем подавление эха
  },
  video: {
    width: { min: 600, ideal: 1280 }, // Минимальная и идеальная ширина видео
    height: { min: 400, ideal: 720 }, // Минимальная и идеальная высота видео
    frameRate: { ideal: 30 }, // Идеальная частота кадров
  },
};

try {
  const constraints = { video: true };
  // [!code highlight]
  const stream = navigator.mediaDevices
    .getDisplayMedia(constraints)
    .then((stream) => {
      videoEl.srcObject = stream;
    });
} catch (error) {
  console.error("Error opening video stream from camera.", error);
}
```

# треки потока

```js
const constraints = { video: true };
const stream = await navigator.mediaDevices.getUserMedia(constraints);
console.log(stream.getTracks());
/* 
[
	{
		"contentHint": "",
		"enabled": true,
		"id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
		"kind": "video",
		"label": "HD-камера FaceTime",
		"muted": false,
		"oncapturehandlechange": null,
		"onended": null,
		"onmute": null,
		"onunmute": null,
		"readyState": "live"
	}
	// <При условии, если мы запросили бы аудио, то тут был бы второй трек>
	// ...
]
*/

// управление

const tracks = stream.getTracks();
// Отключаем трек
tracks[0].enabled = false;
// Включаем трек
tracks[0].enabled = true;

// стоп

stream.getTracks().forEach((track) => track.stop());
```

# Соединение

Основные стадии соединения:

- Сигналинг - соединение между клиентами. Для обмена можно использовать websocket
- создать peer connection - для обмена данных? позволяет обмениваться данными, RTCPeerConnection-интерфейс.

```js
// Конфигурация ICE-серверов
const configuration = {
  iceServers: [
    // STUN серверы
    {
      urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"],
    },
    // TURN сервер
    {
      urls: "turn:turn.example.com:3478",
      username: "username",
      credential: "password",
    },
  ],
};

// Инициализация объекта RTCPeerConnection
const peerConnection = new RTCPeerConnection(configuration);

// Подписываемся на событие icecandidate, которое будет вызываться каждый раз,
// когда новый ICE-кандидат будет найден
peerConnection.addEventListener("icecandidate", (event) => {
  if (event.candidate) {
    // Отправляем ICE-кандидата другому клиенту через сигналинг
    ws.send(clientId, "new-ice-candidate", event.candidate);
  }
});
// на другой стороне

// Конфигурация ICE-серверов
const configuration = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

// Инициализация объекта RTCPeerConnection
const peerConnection = new RTCPeerConnection(configuration);

ws.on("new-ice-candidate", async (message) => {
  if (message.iceCandidate) {
    try {
      // Устанавливаем полученный ICE-кандидат в RTCPeerConnection
      await peerConnection.addIceCandidate(message.iceCandidate);
    } catch (e) {
      console.error("Error adding received ice candidate", e);
    }
  }
});
```

- - ICE - Internet Connection Establishment, подразумевает один из серверов:
- - - STUN (Session Traversal Utilities for NAT) - помогают определить ip порт
- - - TURN (Traversal Using Relays around NAT) - передают данные
- - ICE-кандидатами называют участников соединения

- обмен оферами - для обмена информацией о медиа потоках. SDP (Session Description Protocol) - предложение общаться

```js
const peerConnection = new RTCPeerConnection(configuration);

// Создаем оффер
peerConnection
  .createOffer()
  .then((offer) => {
    // Устанавливаем оффер в качестве локального описания
    return peerConnection.setLocalDescription(offer);
  })
  .then(() => {
    // Отправляем оффер другому клиенту через сигналинг
    ws.send(clientId, "sending-offer", peerConnection.localDescription);
  })
  .catch((error) => {
    console.error("Ошибка при создании оффера:", error);
  });

// принимающая сторона
const peerConnection = new RTCPeerConnection(configuration);

ws.on("send-offer", (offer) => {
  peerConnection
    .setRemoteDescription(offer)
    .then(() => {
      // Создаем ответ на оффер
      return peerConnection.createAnswer();
    })
    .then((answer) => {
      // Устанавливаем ответ в качестве локального описания
      return peerConnection.setLocalDescription(answer);
    })
    .then(() => {
      // Отправляем ответ обратно отправителю оффера
      ws.send(clientId, "sending-answer", peerConnection.localDescription);
    })
    .catch((error) => {
      console.error("Ошибка при обработке оффера:", error);
    });
});
```
