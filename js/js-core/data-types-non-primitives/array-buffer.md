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
