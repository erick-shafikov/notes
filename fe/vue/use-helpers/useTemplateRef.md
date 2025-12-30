# useTemplateRef

позволяет получить ссылку на компонент в шаблоне

```vue
<script setup>
import { useTemplateRef, onMounted } from "vue";

const inputRef = useTemplateRef("input");

onMounted(() => {
  inputRef.value.focus();
});
</script>

<template>
  <input ref="input" />
</template>
```
