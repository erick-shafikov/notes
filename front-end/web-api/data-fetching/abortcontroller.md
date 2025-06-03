# AbortController

Конструктор:

- не принимает параметры

Свойства экземпляра:

- signal - ⇒ AbortSignal

```js
var controller = new AbortController();
var signal = controller.signal;

var downloadBtn = document.querySelector('.download');
var abortBtn = document.querySelector('.abort');

downloadBtn.addEventListener('click', fetchVideo);

abortBtn.addEventListener('click', function() {
  controller.abort();
  console.log('Загрузка прервана');
});

function fetchVideo() {
  ...
  fetch(url, {signal}).then(function(response) {
    ...
  }).catch(function(e) {
    reports.textContent = 'Ошибка загрузки: ' + e.message;
  })
}

```

методы экземпляра:

- abort() - для отмены

# AbortSignal

экземпляр класса возвращает экземпляр AbortController

```js
var controller = new AbortController();
var signal = controller.signal;
```

## статичные методы

- abort(reason) - метод отклонения
- any([]) - передаем итерируемый набор сигналов, сигнал прервется, если один из переданных прервался
- timeout() - вернет AbortSignal, который прервется через определенное время

```js
const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
const result = await res.blob();
```

## свойства экземпляра

- aborted ⇒ boolean
- reason ⇒ reason который передали в AbortSignal.abort(reason)

## методы

- throwIfAborted()

## события

- abort

## обработчики событий

- onabort - когда происходит событие abort
