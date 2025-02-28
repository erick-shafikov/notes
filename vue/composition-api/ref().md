# ref()

используется для примитивных данных

```js
// с помощью setup
import { ref } from "vue";

export default {
  // `setup` это специальный хук, предназначенный для Сomposition API.
  setup() {
    const count = ref(0);

    // передайте состояние шаблону
    return {
      count,
    };
  },
};
```

с помощью script setup

```vue
<script setup>
import { ref } from "vue";

const count = ref(0);

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
    {{ count }}
  </button>
</template>
```
