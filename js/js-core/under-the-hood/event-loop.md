# event loop

- это бесконечный цикл, в котором движок JS ожидает задачи, исполняет их и ждет новые.
- очередь, которую образуют задачи называют macrostack queue.
- Принцип FIFO.
- рендеринг никогда не происходит во время выполнения задачи
- одна отдельная задача может выполняться долго, тогда другие задачи блокируются
- сначала при обработке кода выделяются синхронные задачи, на микрозадачи (промисы, mutationObserver, queueMicrotask()) и макро
  задачи (таймеры и обработчики событий).
- Сначала выполняются все синхронные задачи, которые пришли в очередь, потом выполняются все задачи из микрозадачи, а потом макро задачи

# BP. задача разбития большой задачи

Задача выполняется долго и сразу выведет 999999

```html
<div id="progress"></div>
<script>
  function count() {
    for (let i = 0; i < 1e6; i++) {
      i++;
      progress.innerHTML = i;
    }
  }
  count();
</script>
```

```js
// Пример, когда задача не разбита
let i = 0;
let start = Date.now();
function count() {
  // делаем тяжёлую работу
  for (let j = 0; j < 1e9; j++) {
    i++;
  }
  alert("Done in " + (Date.now() - start) + "ms");
}
count(); //при вызове досчитает и выведет 999…

// с помощью timeout добавляем в очередь очередной вызов функции
let i = 0;
let start = Date.now();
function count() {
  // делаем часть тяжёлой работы (*)
  do {
    i++;
  } while (i % 1e6 != 0); // считает каждый 1 000 000
  if (i == 1e9) {
    // если 100… выведет, что готово
    alert("Done in " + (Date.now() - start) + "ms");
  } else {
    setTimeout(count); // планируем новый вызов (**) если прошел очередной 1 000 000, но не конец, запланировать еще один вызов функции
  }
}
count();
```

```js
// Планирование в начале функции
let i = 0;
let start = Date.now();
function count() {
  // перенесём планирование очередного вызова в начало
  if (i < 1e9 - 1e6) {
    //i меньше последнего шага, при последнем шаге отменит установку таймера
    setTimeout(count); // запланировать новый вызов
  }
  do {
    i++;
  } while (i % 1e6 != 0); //при выполнении вспомнит про назначенный timeout
  if (i == 1e9) {
    alert("Done in " + (Date.now() - start) + "ms");
  }
}
count();
```

Пример с индикатором
изменения i не будут заметны, пока функция не завершится, поэтому мы увидим только последнее значение i:

```html
<div id="progress"></div>
<!-- место для вставки -->
<script>
  function count() {
    for (let i = 0; i < 1e6; i++) {
      i++;
      progress.innerHTML = i;
    }
  }
  count();
</script>

<div id="progress"></div>

<script>
  let i = 0;
  function count() {
    // сделать часть крупной задачи
    do {
      i++;
      progress.innerHTML = i;
    } while (i % 1e3 != 0); //вставлять каждую 1000
    if (i < 1e7) {
      //пока меньше предпоследнего шага – ставить в очередь
      setTimeout(count);
    }
  }
  count();
</script>
```

## сделать что-то после события

```js
menu.onclick = function () {
  // ...
  // создадим наше собственное событие с данными пункта меню, по которому щёлкнули мышью
  let customEvent = new CustomEvent("menu-open", {
    bubbles: true,
  }); // сгенерировать наше событие асинхронно
  setTimeout(() => menu.dispatchEvent(customEvent));
};
```

## Микро и Макро задачи

Микро задачи приходят только из кода. обычно создаются из-за выполнения обработчика промисов .then/catch/finally. Движок выполняет сначала все микро задачи прежде чем выполнить следующую макро задачу
Существует специальная функция queueMicrotask(func), которая помещает функцию в очередь микро задач, так же как и timeout для постановки функции в очередь макрозадач

```js
setTimeout(() => alert("timeout")); //3 очередная макро задача
Promise.resolve().then(() => alert("promise")); //2 из очереди микрозадач
alert("code"); //1 синхронный вызов
```

в данном случае результат будет противоположный timeout

```html
<div id="progress"></div>

<script>
  let i = 0;
  function count() {
    // делаем часть крупной задачи (*)
    do {
      i++;
      progress.innerHTML = i;
    } while (i % 1e3 != 0);
    if (i < 1e6) {
      queueMicrotask(count);
    }
  }
  count();
</script>
```

## Микрозадачи

Обработчики .then/ .catch/ .finally всегда асинхронны. Даже когда промис сразу же выполнение, код в строках ниже .then/ .catch/ .finally будет запущен до этих обработчиков

```js
let promise = Promise.resolve();
promise.then(() => alert("Промис завершен")); // потом вот этот alert
alert("Код выполнен"); // сначала этот alert
```

Происходит это из-за очереди FIFO и выполнение задачи происходит в том случае, если ничего больше не запущено

## BP

```js
// TG
const promise = new Promise((res) => {
  setTimeout(() => res(10), 0);
  res(5);
});

// TG
const myPromise = Promise.resolve(Promise.resolve("Promise"));
function funcOne() {
  myPromise.then((res) => res).then((res) => console.log(res)); //(2) выполняем обещание
  setTimeout(() => console.log("Timeout"), 0); //(5) отправляем в WEB APIб а значит в очереди еще funcTwo
  console.log("Last LIne"); //(1) синхронный вызов
}
async function funcTwo() {
  const res = await myPromise; //так как перед Promise выше стоит await, то ждем выполнения
  console.log(await res); //(3)
  setTimeout(() => console.log("Timeout"), 0); //(6) добавится в WEB API
  console.log("Last Lne"); //(4) синхронный вызов
}
funcOne(); //Last line! Promise! Timeout! если вызвать только funcOne()
funcTwo(); //Promise, Last line, Timeout! если вызвать только funcTwo()
//D: Last line! Promise! Promise! Last line! Timeout! Timeout! если одновременно
```
