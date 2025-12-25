# nextTick

nextTick будет запущен после ре-рендеринга шаблона. Vue проводит обновление DOM асинхронно, они собираются в пачку, что бы убедится что каждый компонент обновится 1 раз

- возвращает промис

Composition API:

```vue
<script setup>
import { nextTick, ref } from "vue";

let counter = ref(0);

async function increment() {
  counter.value++;

  console.log(document.getElementById("counter").textContent); // 0

  await nextTick();
  // DOM is now updated
  console.log(document.getElementById("counter").textContent); // 1
}
</script>

<template>
  <div id="counter">{{ counter }}</div>
  <button @click="increment">+1</button>
</template>
```

Options API:

```vue
<script>
import { nextTick } from "vue";

export default {
  data() {
    return {
      count: 0,
    };
  },
  methods: {
    async increment() {
      this.count++;

      // DOM not yet updated
      console.log(document.getElementById("counter").textContent); // 0

      await nextTick();
      // DOM is now updated
      console.log(document.getElementById("counter").textContent); // 1
    },
  },
};
</script>

<template>
  <button id="counter" @click="increment">{{ count }}</button>
</template>
```
