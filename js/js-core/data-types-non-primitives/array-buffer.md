ArrayBuffer

Объект представляет собой ссылку на поток двоичных данных
Создать можно:

- из Base64
- файла

```js
//length - длина в байтах
new ArrayBuffer(length);
//с полем option
new ArrayBuffer(length, { maxByteLength });
```

# методы

```js
ArrayBuffer.isView(arg); //true если arg на подобии DataView
ArrayBuffer.transfer(oldBuffer, newBufferLength) ;//новый AB с длиной newBufferLength
ArrayBuffer. ;//

```

## методы экземпляра

```js
buffer.resize(newLength); //новый размер
buffer.slice(begin, end); //копия с begin индекса до end
buffer.slice(0); //копия
buffer.transfer(newByteLength); //копия
buffer.transferToFixedLength(newByteLength); //копия с заполнением нулми
```

## свойства экземпляра

```js
buffer.byteLength; //длина
buffer.detached; //был ли изменен
buffer.maxByteLength; //макс длина
buffer.resizable; //масштабируемый ли массив

const buffer1 = new ArrayBuffer(8, { maxByteLength: 16 });
const buffer2 = new ArrayBuffer(8);
console.log(buffer1.resizable); //true
console.log(buffer2.resizable); //false
```

# DataView

предоставляет низко-уровневый интерфейс для чтения и записи различных числовых типов в бинарном ArrayBuffer
