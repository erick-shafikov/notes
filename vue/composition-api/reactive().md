# reactive()

используется только для object, array, map. set

- !нельзя диструктурировать

```js
const raw = {};
const proxy = reactive(raw);

// прокси НЕ РАВЕН оригиналу.
console.log(proxy === raw); // false
```

# связь с ref

```js
const count = ref(0);
const state = reactive({
  count,
});

console.log(state.count); // 0

state.count = 1;
console.log(count.value); // 1
```
