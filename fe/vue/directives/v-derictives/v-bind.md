# v-bind

делает атрибут активным, может брать значения из скрипта

-!!! v-bind:value='some' тоже самое что и :value="some"

```vue
<script>
const name = "John";
</script>

<template>
  <!-- v-bind:value="name" и :value="name" одно и тоже -->
  <input
    v-bind:value="name"
    :value="name"
    v-on:input="name = $event.target.value"
    :disabled="isButtonDisabled"
  />
</template>
```

# объект атрибутов

v-bind и объект аттрибутов

```vue
<script>
const objectOfAttrs = {
  id: "container",
  class: "wrapper",
  style: "background-color:green",
};
</script>

<template>
  <div v-bind="objectOfAttrs"></div>
</template>
<!-- что бы не делать -->
<template>
  <div
    :id="objectOfAttrs.id"
    :class="objectOfAttrs.class"
    :style="objectOfAttrs.style"
  ></div>
</template>
```

# одноименные атрибуты

```vue
<script setup>
const id = "id";
</script>

<template>
  <div :id></div>
</template>
```
