# FILES. Работа с файлами

## ArrayBuffer

Бинарные файлы: ArrayBuffer, Uint8Array, DataView, Blob, File

```js
//длина фиксирована, для доступ к байтам используется специальный метод
//Uint8Array, Uint16Array, Uint32Array, Float64Array
let buffer = new ArrayBuffer(16); // создаётся буфер длиной 16 байт

// Методы

buffer.length; //сколько хранится сейчас
buffer.byteLength; //сколько всего
// Конструктор
new TypedArray(buffer, [byteOffset], [length]);
// buffer

view[0] = 123456;

// теперь пройдёмся по всем значениям
for (let num of view) {
  alert(num); // 123456, потом 0, 0, 0 (всего 4 значения)
}
```

Методы map, slice, find, reduce, нет методов splice, concat

```js
arr.set(fromArr, [offset]);
arr.subarray([begin, end]);
```

## TextDecoder и TextEncoder

Методы которые позволяет преобразовать бинарные файлы в текстовые и наоборот

```js
let decoder = new TextDecoder([label], [options]);
//label - тип кодировки utf-8 по умолчанию
// options {
//  fatal: true,
//  ignoreBOM: true
// }
let str = decoder.decode([input], [options]);
// input - бинарный буфер
// options : {stream: true} - для декодирования потока данных

let encoder = new TextEncoder();

let uint8Array = encoder.encode("Hello");
alert(uint8Array); // 72,101,108,108,111
```

## Blob

Объект, который состоит из:

- необязательной строки type (MIME-тип Multipurpose Internet Mail Extensions) при запросах Content-Type
- blobParts то есть другие Blob, строк и BufferSource

```js
const blob = new Blob(blobParts, options);
// blobParts - Blob, BufferSource, String
// options : {
//   type :MIME-тип,
//   endings: "transparent" | "native" | \r\n | \n
// }
// свойства
blob.size;
blob.type; //MIME
// методы
blob.slice([byteStart], [byteEnd], [contentType]);
const stream = blob.stream();
// для работы с этим методом  stream.getReader(), pipeTo(), tee()
blob.text();
// Создание blob
// из строки
let blob = new Blob(["<html>…</html>"], { type: "text/html" });
let blob = new Blob([hello, " ", "world"], { type: "text/plain" });
// из объекта
const obj = { hello: "world" };
const blob = new Blob([JSON.stringify(obj, null, 2)], {
  type: "application/json",
});
```

Использование с URL

```js
// <a download="hello.txt" href='#' id="link">Загрузить</a>
let blob = new Blob(["Hello, world!"], { type: "text/plain" });
// blob может быть url
link.href = URL.createObjectURL(blob);
// без ссылки
let link = document.createElement("a");
link.download = "hello.txt";
// создается текстовый файл
let blob = new Blob(["Hello, world!"], { type: "text/plain" });
// создаст соответствие URL → Blob
link.href = URL.createObjectURL(blob);
// blob:https://__hostName__/1e67e00e-860d-40a5-89ae-6ab0cbee6273

// удаление ссылки для сборщика мусора
URL.revokeObjectURL(link.href);
```

base 64 представление blob в кодировке ASCII-кодах от 0 до 64

```js
let link = document.createElement("a");
link.download = "hello.txt";

let blob = new Blob(["Hello, world!"], { type: "text/plain" });

// для перевода в base 64 используется FileReader
let reader = new FileReader();
reader.readAsDataURL(blob); // конвертирует Blob в base64 и вызывает onload

reader.onload = function () {
  link.href = reader.result; // url с данными
  link.click();
};
```

Изображение можно переделать в blob с помощью canvas c помощью метод canvas.toBlob
Можно из blob сделать buffer array

```js
let fileReader = new FileReader();

fileReader.readAsArrayBuffer(blob);

fileReader.onload = function (event) {
  let arrayBuffer = fileReader.result;
};
```

## File

Объект file наследуется от Blob

```js
// Создание
new File(fileParts, fileName, [options]);
// fileParts  - Blob BufferSource
// fileName - имя фала
// options: {lastModified : Date} - дата последнего изменения
```

Получаем данные типа File из полей input type file или drag and drop

```js
// <input type="file" onchange="showFile(this)">

function showFile(input) {
  let file = input.files[0]; //так как один файл

  alert(`File name: ${file.name}`); // например, my.png
  alert(`Last modified: ${file.lastModified}`); // например, 1552830408824
}
```

## FileReader

Объект который читает данные из blob

```js
const reader = new FileReader();
// методы
reader.readAsArrayBuffer(blob); //считать как arrayBuffer
reader.readAsText(blob, [encoding]); //считать как текст с encoding кодировкой (utf-8 по умолчанию)
reader.readAsDataURL(blob); //считать данные как base64-кодированный URL.
reader.abort(); //отменить операцию.
// события
reader.addEvenListener("loadstart ", () => {}); // начало чтения
reader.addEvenListener("progress ", () => {}); // срабатывает во время чтения данных.
reader.addEvenListener("load ", () => {}); // нет ошибок, чтение окончено.
reader.addEvenListener("abort ", () => {}); //  вызван abort()
reader.addEvenListener("error ", () => {}); // произошла ошибка.
reader.addEvenListener("loadend ", () => {}); // чтение завершено (успешно или нет).
// после окончания чтения
reader.result; //данные
reader.error; //ошибка
```
