# defineEmits

Позволяет обрабатывать пробрасываемые коллбеки в компоненты

```vue
<!-- корневой компонент -->
<script setup>
import { ref } from "vue";
import CitySelect from "./components/CitySelect.vue";

// реактивное значение
let savedCity = ref("Moscow");

// коллбек на изменение
const getCity = async (city) => {
  savedCity.value = city;
};
</script>

<template>
  <main class="main">
    <!-- отображение реактивного значения -->
    {{ savedCity }}
    <!-- дочерний компонент -->
    <!-- selectCity в дочернем в атрибутах можно через дефис -->
    <CitySelect @select-city="getCity" />
  </main>
</template>
```

```vue
<!-- дочерний компонент -->
<script setup>
import { ref } from "vue";

// вариант 1 через массив принятых коллбеков
const emit = defineEmits(["selectCity"]);

//вариант 2 через объект с коллбеками
const emit = defineEmits({
  // в родительском компоненте сигнатура (arg1)=>void
  selectCity(payload) {
    return payload;
  },
});

// коллбек на изменение значений
function select() {
  // первый аргумент - событие, второй параметр
  emit("selectCity", "london");
}
</script>

<template>
  <button @click="select()">Сохранить</button>
</template>
```

Типизация

```ts
const emit = defineEmits<{
  (e: "change", id: number): void;
  (e: "update", value: string): void;
}>();

// 3.3+: alternative, more succinct syntax
const emit = defineEmits<{
  change: [id: number]; // named tuple syntax
  update: [value: string];
}>();
```
