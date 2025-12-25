# reactive

используется только для object, array, map, set

- !нельзя деструктурировать

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

ref + map + array

```js
const books = reactive([ref("Vue 3 Guide")]);
// need .value here
console.log(books[0].value);

const map = reactive(new Map([["count", ref(0)]]));
// need .value here
console.log(map.get("count").value);
```
