# Timers

## setTimeout

позволяет вызвать функцию один раз через определённый интервал времени ← индикатор времени
let timerId = setTimeout(func|code, [delay], [arg1], [arg2], ...)
func|code - Функция или строка кода для выполнения. Обычно это функция. По историческим причинам можно передать и строку кода, но это не рекомендуется.
delay - Задержка перед запуском в миллисекундах (1000 мс = 1 с). Значение по умолчанию – 0.
arg1, arg2… Аргументы, передаваемые в функцию (не поддерживается в IE9-)

```js
// !!! добавляя скобки () после функции:
setTimeout(sayHi(), 1000); // не правильно!
// Пример:
function sayHi(phrase, who) {
  alert(phrase + "," + sho);
}
setTimeout(sayHI, 1000, "Привет", "Джон");
```

Первый аргумент может быть и строкой:

```js
setTimeout("alert("Привет")", 1000);
```

Отмена через clearTimeout.

```js
// Синтаксис для отмены:
let timerId = setTimeout();
clearTimeout(timerId);

let timerId = setTimeout(() => alert("nothing"), 1000);
alert(timerId); // какое-то рандомные число  в виде идентификатора  clearTimeout(timerId);
alert(timerId); // после отмены не принимает значение  null
```

## setInterval

setInterval – функция запускается не один раз
let timerId = setInterval(func|code.[delay], [arg], [arg2])

для того чтобы остановить нужно вызвать clearInterval(timerId)

```js
let timerId = setInterval(() => alert("tick"), 2000);

setTimeout(() => {
  () => {
    cleanInterval(timerId);
    alert("stop");
  },
    5000;
}); //во время показа alert время тоже идет
```

## рекурсивный setTimeout

```js
let timerId = setTimeout(function tick() {
  alert("tick");
  timerId = setTimeout(tick, 2000); // (*)
}, 2000); // let timerId = setTimeout(function(){timerId = setTimeout(function(), 2000)}, 2000);

// Или

let delay = 5000;
let timerId = setTimeout(function request() {
  // ...отправить запрос...
  if (condition) {
    /*ошибка запроса из-за перегрузки сервера*/
    // увеличить интервал для следующего запроса  delay *= 2;
  }
  timerId = setTimeout(request, delay);
}, delay);

// другими словами:
let timerId = setTimeout(function func() {
  //какая –то функция  timerId = setTimeout(func, [delay])}, [delay])
});
```

# BP

## debounce

```js
const debounce = (callback, interval = 0) => {
  let prevTimeoutId;

  return (...args) => {
    clearTimeout(prevTimeoutId);
    prevTimeoutId = setTimeout(() => callback(...args), interval);
  };
};
```

## задачи

```js
// TG setTimeout и цикл
for (let i = 0; i < 10; i++) {
  setTimeout(function () {
    alert(i);
  }, 100); //1,2,3.....9
}
```
