- !!! шаблоны могут быть частью шаблона

```vue
<script setup>
const some_class = "id";
</script>

<template>
  <div :class="`color-${some_class}`"></div>
</template>
```
