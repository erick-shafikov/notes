Atomics

- для работы с SharedArrayBuffer.
- не вызывается с new, все методы статические

# Atomics.add()

добавляет значение к текущему по указанной позиции в массиве и возвращает предыдущее значение в этой позиции

```js
var sab = new SharedArrayBuffer(1024);
var ta = new Uint8Array(sab);

Atomics.add(ta, 0, 12); // возвращает 0, предыдущее значение
Atomics.load(ta, 0); // 12
```

# Atomics.and()

# Atomics.compareExchange()

# Atomics.exchange()

# Atomics.load()

# Atomics.or()

# Atomics.store()

# Atomics.sub()

# Atomics.xor()
