@ - привязка к событию, кратка запись для v-on

# @-директивы событий

```vue
<!-- родительский компонент -->
<script setup>
import CitySelect from "./components/CitySelect.vue";

const getCity = (city) => {
  console.log(city);
};
</script>

<template>
  <!-- так как вызывается событие selectCity -->
  <CitySelect @select-city="getCity" />
</template>
```

```vue
<!-- дочерний компонент -->
<script setup>
// вариант номер 1 в виде массива
// событие превратиться в атрибут @select-city
const emit = defineEmits(["selectCity"]);

// вариант номер 2 в виде объекта валидации
// selectCity: null - нет валидации
  selectCity(payload) { //функция валидации
    // логика валидации
    // если вернем false то будет warning в консоли
    console.log("validating payload");
    return payload;
  },

function select() {
  // первый аргумент - название selectCity или select-city разницы нет
  // London - аргумент, можно передать несколько аргументов
  emit("select-city", "London");
}
</script>

<template>
  <!-- вызов на целевом компоненте -->
  <Button @click="select()">Изменить город</Button>
</template>
```

# @click

```html
<a v-on:click="doSomething"> ... </a>

<!-- shorthand -->
<a @click="doSomething"> ... </a>
```

```vue
<script>
const add = () => {};
</script>
<!-- add из methods -->
<button type="button" class="btn btn-primary" @click="add">Add number</button>
```

# @submit

```vue
<script setup>
sendForm;
</script>

<template>
  <form v-if="!formDone" @submit.prevent="sendForm"></form>
</template>
```

# @scroll

TODO

# .prevent

TODO
