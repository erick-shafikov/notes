# defineProps

на js

```vue
<script>
const { label, stat } = defineProps(["label", "stat"]);

// или через конструктор
const {
  //значение по умолчанию
  label = "Не задано",
  stat,
} = defineProps({
  label: String,
  stat: String,
});
</script>

<template>
  <div class="stat">
    <div class="stat-name">{{ label }}</div>
    <div class="stat-value">{{ stat }}</div>
  </div>
</template>
```

на ts

```ts
const { foo, bar } = defineProps<{
  foo: string;
  bar?: number;
}>();
```

# withDefaults (deprecated)

```vue
<script>
// deprecated
// раньше нельзя было пользоваться деструктуризацией
const props = withDefaults(
  defineProps({
    label: String,
    stat: String,
  }), {label: Не задано}
);
</script>

<template>
  <div class="stat">
    <div class="stat-name">{{ label }}</div>
    <div class="stat-value">{{ stat }}</div>
  </div>
</template>
```

на ts

```ts
interface Props {
  msg?: string;
  labels?: string[];
}

const props = withDefaults(defineProps<Props>(), {
  msg: "hello",
  labels: () => ["one", "two"],
});
```
