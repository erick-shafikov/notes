# template

Позволяет обернуть контент в пустой контейнер

допустимые выражения в шаблоне

```vue
<template>
  <div>{{ number + 1 }}</div>
  <div>{{ ok ? "YES" : "NO" }}</div>
  <div>{{ message.split("").reverse().join("") }}</div>

  <div :id="`list-${id}`"></div>
</template>
```

нельзя

```vue
<template>
  {{ var a = 1 }}

  {{ if (ok) { return message } }}
  <!-- будет вызываться каждый раз при обновлении -->
  {{ formatDate(date) }}
</template>
<script></script>
```
