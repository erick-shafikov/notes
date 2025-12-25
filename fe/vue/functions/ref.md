# ref()

используется для реактивных данных. Значение мутабельное по полю value. В шаблоне поле value достается автоматически

```vue
<script setup>
import type { Ref } from "vue";
import { ref } from "vue";

// типизация
const count: Ref<string | number> = ref(0);

/* 
// псевдокод, а не реальная реализация
const myRef = {
  _value: 0,
  get value() {
    track()
    return this._value
  },
  set value(newValue) {
    this._value = newValue
    trigger()
  }
}
*/

function increment() {
  count.value++;
}
</script>

<template>
  <button @click="increment">
    <!-- можно и так, без value -->
    <!-- <button @click="count++"> -->
    {{ count }}
  </button>
</template>
```

```js
// с помощью setup
import { ref } from "vue";

export default {
  setup() {
    const count = ref(0);

    // передайте состояние шаблону
    return {
      count,
    };
  },
};
```
