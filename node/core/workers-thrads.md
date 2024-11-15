```js
//isMainThread - проверить не в основном ли треде
const { Worker, isMainThread } = require("worker_threads");

const firstWorker = new Worker("cpu_intensive.js");
const secondWorker = new Worker("some eval code", { eval: true });
```

```js
const {
  Worker,
  isMainThread,
  // event emitter для main thread
  parentPort,
  // позволяет предавать данные 3.1[3]
  workData,
} = require("worker_threads");

if (isMainThread) {
  const worker = new Worker(__filename, {
    // можем добавить разные параметры 3.2[3]
    workerData: {
      param1: "param 1",
      param2: "param 2",
    },
  });

  // получить сообщение из побочного потока 1.1[2]
  worker.on("message", (msg) => {
    console.log(msg);
  });

  // отправить сообщение в побочный поток 2.1[2]
  worker.postMessage("to second thread");
} else {
  // Отправить сообщение в главный поток 1.2[2]
  parentPort.message("start");
  someHeaveOperation();
  parentPort.message("end");

  //получить сообщение из главного потока 2.2[2]
  parentPort.on("message", (msg) => console.log("from main thread"));
  //получить сообщение из главного потока 3.3[3]
  parentPort.on("message", () => {
    console.log(`${workerData.param1}`);
    console.log(`${workerData.param2}`);
  });
}

const someHeaveOperation = () => {
  //функция которая требует много вычислительных мощностей
};
```
