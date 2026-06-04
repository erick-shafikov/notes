# SharedArrayBuffer

## Назначение

`SharedArrayBuffer` — общий участок памяти, который может одновременно использоваться несколькими потоками (`Worker`, `Web Worker`, `Node.js Worker Threads`).

В отличие от `ArrayBuffer`, при передаче между потоками память не копируется и не переносится — все потоки работают с одним и тем же буфером.

Основное применение:

- многопоточность
- обмен данными между Worker'ами
- lock-free алгоритмы
- высокопроизводительные вычисления
- реализация разделяемых очередей и кешей

## Создание

### Конструктор

```js
const sab = new SharedArrayBuffer(length);
```

### Параметры

| Параметр | Тип    | Описание               |
| -------- | ------ | ---------------------- |
| length   | number | Размер буфера в байтах |

Пример:

```js
// Создаст общий буфер размером 1024 байта.
const sab = new SharedArrayBuffer(1024);
```

## Свойства экземпляра

### byteLength

Размер буфера в байтах.

```js
const sab = new SharedArrayBuffer(1024);

console.log(sab.byteLength);
// 1024
```

---

## Методы экземпляра

### slice(begin, end)

Создает новый `SharedArrayBuffer`, содержащий копию части данных.

```js
const sab = new SharedArrayBuffer(16);

const copy = sab.slice(0, 8);

console.log(copy.byteLength);
// 8
```

Важно:

- данные копируются
- связь между буферами отсутствует

## Работа через TypedArray

Сам по себе `SharedArrayBuffer` не хранит значения.

Для чтения/записи нужен TypedArray:

```js
const sab = new SharedArrayBuffer(4);

const arr = new Int32Array(sab);

arr[0] = 100;

console.log(arr[0]);
// 100
```

Часто используются:

- Int8Array;
- Uint8Array;
- Uint16Array;
- Uint32Array;
- Int16Array;
- Int32Array;
- Float32Array;
- Float64Array;
- BigInt64Array;
- BigUint64Array;

## Совместное использование между Worker

```js
// main.js
const worker = new Worker("worker.js");

const sab = new SharedArrayBuffer(4);

worker.postMessage(sab);

const data = new Int32Array(sab);

data[0] = 42;

// worker.js
onmessage = ({ data: sab }) => {
  const arr = new Int32Array(sab);

  console.log(arr[0]);
};
```

Worker увидит значение `42`.

Передача происходит без копирования памяти.

## Атомарные операции (Atomics)

При одновременной записи из нескольких потоков возможны race condition.

Для синхронизации используется объект `Atomics`.

Пример:

```js
const sab = new SharedArrayBuffer(4);
const arr = new Int32Array(sab);

Atomics.store(arr, 0, 10);

console.log(Atomics.load(arr, 0));
// 10
```

Инкремент без гонок:

```js
Atomics.add(arr, 0, 1);
```

Часто используемые методы Atomics

```js
// Чтение значения.
Atomics.load(arr, index);
// Запись значения.
Atomics.store(arr, index, value);
// Атомарное сложение.
Atomics.add(arr, index, value);
//Атомарное вычитание.
Atomics.sub(arr, index, value);
// Побитовое AND.
Atomics.and(arr, index, value);
// Побитовое OR.
Atomics.or(arr, index, value);
// Побитовое XOR.
Atomics.xor(arr, index, value);
// Заменяет значение и возвращает старое.
Atomics.exchange(arr, index, value);
//CAS-операция.
Atomics.compareExchange(arr, index, expected, replacement);
// Если значение равно `10`, станет `20`.
Atomics.compareExchange(arr, 0, 10, 20);
// Блокирует Worker до изменения значения.
Atomics.wait(arr, 0, 0);
// Будит ожидающие потоки.
Atomics.notify(arr, 0, 1);
```

Работает только для:

- Int32Array;
- BigInt64Array;

Не работает в главном браузерном потоке.

---

## Отличия от ArrayBuffer

| ArrayBuffer                                | SharedArrayBuffer           |
| ------------------------------------------ | --------------------------- |
| память не разделяется                      | память разделяется          |
| при передаче обычно копируется/переносится | используется всеми потоками |
| синхронизация не нужна                     | нужен Atomics               |
| безопаснее                                 | возможны race condition     |

---

## Ограничения браузеров

Для использования в браузере обычно требуется:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Иначе создание может быть запрещено из-за защиты от Spectre.

Проверка:

```js
if (crossOriginIsolated) {
  const sab = new SharedArrayBuffer(1024);
}
```

---

## Кратко

```js
const sab = new SharedArrayBuffer(1024);

const arr = new Int32Array(sab);

arr[0] = 123;

Atomics.add(arr, 0, 1);

console.log(arr[0]);
// 124
```

`SharedArrayBuffer` = общая память между потоками.
`TypedArray` = доступ к данным.
`Atomics` = безопасная синхронизация.
