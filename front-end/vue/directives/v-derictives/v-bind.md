# v-bind

делает атрибут активным, может брать значения из скрипта

тоже самое что и :value="some"

```vue
<template>
  <input
    type="text"
    v-bind:value="name"
    v-on:input="name = $event.target.value"
    :disabled="isButtonDisabled"
  />
</template>

<script>
export default {
  data() {
    return {
      name: "",
    };
  },
};
</script>
```

v-bind и объект аттрибутов

```vue
<template>
  <div v-bind="objectOfAttrs"></div>
</template>

<script>
const objectOfAttrs = {
  id: "container",
  class: "wrapper",
  style: "background-color:green",
};
</script>
```

<!--  -->

```vue
<template></template>

<script></script>
```
