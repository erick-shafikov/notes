# computed

нужны для вычисления над reactive. На вход функция принимает коллбек, который возвращает вычисленное значение. Первым параметром в коллбек идет предыдущее состояние.

Разница между обычными хелпер функциями и computed - кешированние

```vue
<script setup>
import { reactive, computed } from "vue";

const author = reactive({
  name: "John Doe",
  books: [
    "Vue 2 - Advanced Guide",
    "Vue 3 - Basic Guide",
    "Vue 4 - The Mystery",
  ],
});

// ref вычисляемого свойства
const publishedBooksMessage = computed(() => {
  return author.books.length > 0 ? "Да" : "Нет";
});
</script>

<template>
  <p>Есть опубликованные книги:</p>
  <span>{{ publishedBooksMessage }}</span>
  <!-- можно и так, но не произойдет кеширования -->
  <span>{{ publishedBooksMessage }}</span>
</template>
```

# установка значений с помощью сеттера

```vue
<script setup>
import { ref, computed } from "vue";

const firstName = ref("John");
const lastName = ref("Doe");

const fullName =
  computed <
  SomeType >
  {
    // геттер (для получения значения)
    get() {
      return firstName.value + " " + lastName.value;
    },
    // сеттер (при присвоении нового значения)
    // должны быть без побочных эффектов
    set(newValue) {
      // Примечание: это синтаксис деструктурирующего присваивания
      [firstName.value, lastName.value] = newValue.split(" ");
    },
    //для дебага
    onTrack(e) {
      // triggered when count.value is tracked as a dependency
      debugger;
    },
    onTrigger(e) {
      // triggered when count.value is mutated
      debugger;
    },
  };

fullName.value = "Ivan Ivanov";
</script>
```
