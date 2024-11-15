```js
// создание
const EventEmitter = require("events");

let myEventEmitter = new EventEmitter();

// регистрация события 1
myEventEmitter.on("event1", () => {
  console.log("event1");
});

// регистрация события c аргументами
myEventEmitter.on("event-with-params", (params) => {
  console.log(params);
});

myEventEmitter.emit("event1"); //"event1"
myEventEmitter.emit("event-with-params", "some params"); //some params'
```
