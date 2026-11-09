# event loop

У каждого [агента](./execution-model.md#agent) свой event-loop. window el- привзяан к документу (в свободное время requestIdleCallback), worker el - нет idle состояния, worklet el

- алгоритм браузера на потоке агента, бесконечный цикл, в котором движок JS ожидает задачи, исполняет их и ждет новые
- Принцип FIFO.

Цикл основывается на task'ах - условная единица, любая операция, состоит:

- steps - что сделать
- source - ее источник, откуда пришла задача (user interaction, networking, DOM manipulation, rendering, остальное апи - таймеры и так далее)
- document - к какому документу относится
- script evaluation environment settings object set

живут task в task queue, раскладывает из источник source

Цикл для task:

- одна отдельная задача может выполняться долго, тогда другие задачи блокируются, берется task'а
- сначала при обработке кода выделяются синхронные задачи (Call Stack), на микрозадачи (Task Queue (macrotasks)) (промисы, mutationObserver, queueMicrotask()) и макро задачи (Microtask Queue (микрозадачи)) (таймеры и обработчики событий).
- Сначала выполняются все синхронные задачи, которые пришли в очередь, потом выполняются все задачи из микрозадачи, а потом макро задачи
- рендеринг (в браузере) никогда не происходит во время выполнения задачи
- цикл повторяется

```html
<!-- Задача выполняется долго и сразу выведет 999999 -->
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

# Микро и Макро задачи

Микро задачи - от алгоритма queue a microtask. Микротаски и макротаски лежат в разных очередях. На самом деле никакой второй очереди нет

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

- очередь микрозадач - это set, не очередь

Обработчики .then/ .catch/ .finally всегда асинхронны. Даже когда промис сразу же выполнение, код в строках ниже .then/ .catch/ .finally будет запущен до этих обработчиков

```js
let promise = Promise.resolve();
promise.then(() => alert("Промис завершен")); // потом вот этот alert
alert("Код выполнен"); // сначала этот alert
```

Происходит это из-за очереди FIFO и выполнение задачи происходит в том случае, если ничего больше не запущено

## spin и pause event loop

если нужно подождать среди таски (для случаев парсинга скриптов)

- запомнить task
- скопировать стек
- запустить checkpoint
- выполнения условия ожидания

Pause - блокировка, при использовании alert, confirm, prompt, print() и диалог beforeunload

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

```js
// Что выведется в консоль?
(function () {
  console.log(1); // 1
  setTimeout(() => console.log(2), 1000); // 7
  setTimeout(() => console.log(3), 0); // 6
  Promise.resolve(5).then(console.log); // 4
  new Promise((resolve) => {
    console.log(6); // 2
    resolve(7);
  }).then(console.log); // 5
  console.log(4); // 3
})();
```

### requestAnimationFrame и requestIdleCallback

Место в цикле: одна макрозадача → все микрозадачи → (если пора рисовать кадр, ~16.6ms при 60Hz) rAF-колбеки → style → layout → paint → (если до следующего кадра осталось время) rIC-колбеки

- rAF — перед отрисовкой следующего кадра, не очередь задач
- rIC — в простое, после отрисовки, только если очередь задач пуста
- после каждого rAF/rIC колбека выполняются все микрозадачи

```js
// Порядок всех видов задач
console.log(1);
setTimeout(() => console.log("timeout"), 0);
requestAnimationFrame(() => console.log("raf"));
requestIdleCallback(() => console.log("idle"));
Promise.resolve().then(() => console.log("promise"));
queueMicrotask(() => console.log("microtask"));
console.log(2);
// 1 2 promise microtask | timeout raf idle
// timeout vs raf - порядок НЕ гарантирован: зависит от того, в какой момент кадра запущен код
// (если кадр вот-вот рисуется - raf может быть раньше timeout)
// idle всегда последний - пока в очереди есть timeout, браузер не "простаивает"
```

```js
// Микрозадачи выполняются между rAF колбеками
requestAnimationFrame(() => {
  console.log(1);
  Promise.resolve().then(() => console.log(2));
});
requestAnimationFrame(() => {
  console.log(3);
});
// 1 2 3 - после каждого колбека чекпоинт микрозадач, оба колбека в одном кадре
```

```js
// rAF внутри rAF - выполнится в следующем кадре, а не в текущем
requestAnimationFrame((t1) => {
  console.log("frame 1");
  requestAnimationFrame((t2) => {
    console.log("frame 2", t2 - t1); // ~16.6
  });
});

// все rAF одного кадра получают одинаковый timestamp
requestAnimationFrame((t) => console.log(t));
requestAnimationFrame((t) => console.log(t)); // то же самое число
```

```js
// Классика: анимация не с той позиции
// хотим: box прыгает в 1000px, затем плавно едет в 500px
box.style.transform = "translateX(1000px)";
box.style.transition = "transform 1s ease-in-out";
box.style.transform = "translateX(500px)";
// поедет 0 → 500: стили считаются только при рендере, промежуточное значение 1000 браузер не увидел

// один rAF не поможет - колбек выполнится ДО расчёта стилей того же кадра
box.style.transform = "translateX(1000px)";
requestAnimationFrame(() => {
  box.style.transition = "transform 1s ease-in-out";
  box.style.transform = "translateX(500px)"; // всё ещё 0 → 500
});

// решение 1: двойной rAF - первый кадр отрисует 1000px, во втором ставим переход
box.style.transform = "translateX(1000px)";
requestAnimationFrame(() => {
  requestAnimationFrame(() => {
    box.style.transition = "transform 1s ease-in-out";
    box.style.transform = "translateX(500px)"; // 1000 → 500
  });
});

// решение 2: принудительный reflow - браузер синхронно считает стили
box.style.transform = "translateX(1000px)";
getComputedStyle(box).transform; // или box.offsetWidth
box.style.transition = "transform 1s ease-in-out";
box.style.transform = "translateX(500px)"; // 1000 → 500
```

```js
// Сколько раз выполнится за секунду?
let timeouts = 0;
let frames = 0;
const start = performance.now();

(function tick() {
  timeouts++;
  if (performance.now() - start < 1000) setTimeout(tick, 0);
  else console.log("timeouts", timeouts); // ~200-250: после 5 вложенных вызовов задержка ≥ 4ms
})();

(function frame() {
  frames++;
  if (performance.now() - start < 1000) requestAnimationFrame(frame);
  else console.log("frames", frames); // ~60 (частота экрана: 120 на 120Hz)
})();
// в фоновой вкладке rAF ставится на паузу (frames ≈ 0), setTimeout троттлится до ≥ 1000ms
```

```js
// Анимация на setTimeout vs rAF
// setTimeout(16) не синхронизирован с кадрами: часть кадров пропускается, часть рисует 2 шага - рывки
let x = 0;
function moveTimeout() {
  x += 2;
  box.style.transform = `translateX(${x}px)`;
  if (x < 500) setTimeout(moveTimeout, 16);
}

// rAF: ровно 1 шаг на кадр, скорость считаем от времени, а не от количества кадров
let startTime;
function moveRaf(t) {
  startTime ??= t;
  const progress = Math.min((t - startTime) / 1000, 1); // 1s на всю анимацию
  box.style.transform = `translateX(${progress * 500}px)`;
  if (progress < 1) requestAnimationFrame(moveRaf);
}
requestAnimationFrame(moveRaf);
```

```js
// Блокировка: rAF не выполнится, пока занят стек
requestAnimationFrame(() => console.log("raf"));
setTimeout(() => console.log("timeout"), 0);
const end = Date.now() + 1000;
while (Date.now() < end) {} // 1s синхронной работы - ни кадров, ни таймеров
console.log("sync");
// sync | timeout raf (или raf timeout) - через ~1s, кадры за это время пропущены
```

```js
// rIC: дробление неважной работы (аналитика, предзагрузка) по свободному времени кадра
const tasks = Array.from({ length: 1000 }, (_, i) => () => sendLog(i));

function work(deadline) {
  // timeRemaining() - сколько ms осталось до следующего кадра (максимум 50)
  // didTimeout - true, если колбек вызван по истечении timeout, а не в простое
  while (
    (deadline.timeRemaining() > 0 || deadline.didTimeout) &&
    tasks.length
  ) {
    tasks.shift()();
  }
  if (tasks.length) requestIdleCallback(work, { timeout: 2000 });
}
requestIdleCallback(work, { timeout: 2000 });
// без timeout при постоянной загрузке (анимации, тяжёлые таймеры) колбек может не вызваться никогда
```

```js
// rIC и DOM: менять DOM в rIC не стоит - кадр уже отрисован, изменения вызовут
// незапланированный layout и могут не уложиться в timeRemaining
requestIdleCallback(() => {
  const data = prepareData(); // подготовка - ок
  requestAnimationFrame(() => render(data)); // DOM - в следующий кадр
});
```

```js
// Порядок нескольких rIC и вложенных
requestIdleCallback(() => {
  console.log("idle 1");
  requestIdleCallback(() => console.log("idle 3")); // в следующем периоде простоя
  Promise.resolve().then(() => console.log("micro"));
});
requestIdleCallback(() => console.log("idle 2"));
// idle 1 micro idle 2 | idle 3
```

## рекурсивные вставки в el

```js
// стандартная задача 1 4 6 | 3 | 2 5
console.log(1);

setTimeout(function () {
  console.log(2);
});

Promise.resolve(3).then(console.log);

console.log(4);

setTimeout(function () {
  console.log(5);
}, 0);

console.log(6);
```

```js
// рекурсивно вызванная 1 4 6 | foo1 3 foo1 foo1 ...
console.log(1);

setTimeout(function () {
  console.log(2);
});

Promise.resolve(3).then(console.log);

console.log(4);

setTimeout(function () {
  console.log(5);
}, 0);

console.log(6);

const foo1 = () => {
  console.log("foo1");

  // поставит в очередь мт foo1, которая вызовет foo1, которая опять foo1
  return Promise.resolve().then(foo1);

  // вызовет foo1
  foo1();
};

// первый вызов
foo1();
```

```js
// рекурсивно вызванная 1 4 6 | foo2 3 4 5 foo2 foo2 ...
console.log(1);

setTimeout(function () {
  console.log(2);
});

Promise.resolve(3).then(console.log);

console.log(4);

setTimeout(function () {
  console.log(5);
}, 0);

console.log(6);

const fool2 = () => {
  console.log("foo2");

  setTimeout(foo2);
};

foo2();
```
