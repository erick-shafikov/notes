# ArrayBuffer

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

Бинарные файлы: ArrayBuffer, Uint8Array, DataView, Blob, File

```js
//длина фиксирована, для доступ к байтам используется специальный метод
//Uint8Array, Uint16Array, Uint32Array, Float64Array
let buffer = new ArrayBuffer(16); // создаётся буфер длиной 16 байт

// Методы

// Конструктор
new TypedArray(buffer, [byteOffset], [length]);
// buffer

view[0] = 123456;

// теперь пройдёмся по всем значениям
for (let num of view) {
  alert(num); // 123456, потом 0, 0, 0 (всего 4 значения)
}
```

# методы

```js
ArrayBuffer.isView(arg); //true если arg на подобии DataView
ArrayBuffer.transfer(oldBuffer, newBufferLength) ;//новый AB с длиной newBufferLength
ArrayBuffer. ;//

```

## методы экземпляра

Методы map, slice, find, reduce, нет методов splice, concat

```js
arr.set(fromArr, [offset]);
arr.subarray([begin, end]);
```

```js
buffer.resize(newLength); //новый размер
buffer.slice(begin, end); //копия с begin индекса до end
buffer.slice(0); //копия
buffer.transfer(newByteLength); //копия
buffer.transferToFixedLength(newByteLength); //копия с заполнением нулми
```

## свойства экземпляра

```js
buffer.length; //сколько хранится сейчас
buffer.byteLength; //сколько всего
buffer.detached; //был ли изменен
buffer.maxByteLength; //макс длина
buffer.resizable; //масштабируемый ли массив

const buffer1 = new ArrayBuffer(8, { maxByteLength: 16 });
const buffer2 = new ArrayBuffer(8);
console.log(buffer1.resizable); //true
console.log(buffer2.resizable); //false
```

# BigInt64Array

- 64битные знаковые

# BigUint64Array

- 64битные без знаковые

# DataView

предоставляет низко-уровневый интерфейс для чтения и записи различных числовых типов в бинарном ArrayBuffer

```js
const littleEndian = (() => {
  const buffer = new ArrayBuffer(2);
  new DataView(buffer).setInt16(0, 256, true /* littleEndian */);

  // Int16Array использует порядок байтов платформы.
  return new Int16Array(buffer)[0] === 256;
})();

console.log(littleEndian); // true или false
```

- getBigInt64(), getBigUint64(), getFloat32(), getFloat64(), getInt16(), getInt32(), getInt8(), getUint16(), getUint32(), getUint8() - методы которые считывают 2,4,8 байт и интерпретируют как 8, 16..
- setBigInt64(), setBigUint64(), setFloat32(), setFloat64(), setInt16(), setInt32(), setInt8(), setUint16(), setUint32(), setUint8() устанавливают

# Float16Array (-chrome, -edge)

массив 16битных с плавающей точкой

# Float32Array

массив 32битных чисел с плавающей точкой

# Float64Array

64битные с плавающей точкой float

# Int16Array

# Int32Array

# Int8Array

# Uint16Array

# Uint32Array

# Uint8Array
