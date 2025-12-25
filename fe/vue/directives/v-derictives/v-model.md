Создает двустороннее связывание

Можно использовать только на

- input
- select
- textarea
- компоненты

с модификаторами:

- .lazy - отслеживание события change вместо input
- .number - приведение корректной строки со значением к числу
- .trim - удаление пробелов в начале и в конце строки

v-model распадается на две детективы

на @onChange и на v-data

```vue
<template>
  <input v-model="name" />

  <input v-bind:value="name" v-on:input="name = $event.target.value" />
</template>
```

в данном приме без вложенности компонентов

```vue
<script>
let name = ref("");
</script>

<template>
  <div class="wrapper">
    <div class="sample">
      <input type="text" v-model="name" />
      <hr />
      <h2>Hello, {{ name }}</h2>
    </div>
  </div>
</template>
```

## c чекбоксами

```vue
<script setup>
import { ref } from "vue";

const checkedNames = ref([]);
</script>
<template>
  <input type="checkbox" id="checkbox" v-model="checked" />
  <!-- будут добавляться в массив -->
  <label for="checkbox">{{ checked }}</label>

  <div>Отмеченные имена: {{ checkedNames }}</div>

  <input type="checkbox" id="jack" value="Jack" v-model="checkedNames" />
  <label for="jack">Jack</label>

  <input type="checkbox" id="john" value="John" v-model="checkedNames" />
  <label for="john">John</label>

  <input type="checkbox" id="mike" value="Mike" v-model="checkedNames" />
  <label for="mike">Mike</label>
  <!-- при значениях для false и true -->
  <input type="checkbox" v-model="toggle" true-value="да" false-value="нет" />
</template>
```

## c радо кнопками

```vue
<template>
  <div>Выбрано: {{ picked }}</div>

  <input type="radio" id="one" value="Один" v-model="picked" />
  <label for="one">Один</label>

  <input type="radio" id="two" value="Два" v-model="picked" />
  <label for="two">Два</label>
</template>
```

## c селектом

```vue
<template>
  <div>Выбраны: {{ selected }}</div>

  <select v-model="selected" multiple>
    <option>А</option>
    <option>Б</option>
    <option>В</option>
  </select>
</template>
```

## multiline

```vue
<template>
  <!-- white-space -->
  <p style="white-space: pre-line;">{{ message }}</p>
  <textarea v-model="message" placeholder="add multiple lines"></textarea>
</template>
```

# проброс в компоненты

```vue
<!-- родительский компонент -->
<script setup>
let city = ref("Moscow");
</script>
<template>
  <Input placeholder="Введите город" v-model="city" />
</template>
```

```vue
<!-- дочерний компонент -->
<script setup>
// v1 2-way binding
// const emit = defineEmits(["update:value"]);
// const props = defineProps(["value"]);
const data = defineModel({
  type: String,
  required: true,
  default: "default value",
});
</script>
<template>
  <!-- <input
    class="input"
    @input="emit('update:value', $event.target.value)"
    :value="props.value"
  /> -->
  <!-- v2 model -->
  <!-- <input class="input" @input="model = $event.target.value" :value="model" /> -->
  <input class="input" v-model="data" />
</template>
```

# c аргументами

```vue
<script setup>
const title = defineModel("title");
</script>

<template>
  <input type="text" v-model="title" />
</template>
```

# несколько v-model для дочернего компонента

```vue
<script setup>
let first = ref("");
let last = ref("");
</script>
<template>
  <!-- каждое отдельно состояние можно пробросить через аргумент -->
  <UserName v-model:first-name="first" v-model:last-name="last" />
</template>
```

```vue
<!-- UserName -->
<script setup>
const firstName = defineModel("firstName");
const lastName = defineModel("lastName");
</script>

<template>
  <input type="text" v-model="firstName" />
  <input type="text" v-model="lastName" />
</template>
```

# кастомные модификаторы

defineModel возвращает вторым аргументов модификаторы переданные в set

```vue
<script setup>
const [model, modifiers] = defineModel({
  set(value) {
    if (modifiers.capitalize) {
      return value.charAt(0).toUpperCase() + value.slice(1);
    }
    return value;
  },
});
</script>

<template>
  <input type="text" v-model="model" />
</template>
```

# как работает

Под капотом в родительском компоненте

```vue
<script setup></script>
<template>
  <Child :modelValue="foo" @update:modelValue="($event) => (foo = $event)" />
</template>
```

Под капотом в дочернем компоненте

```vue
<script setup>
const props = defineProps(["modelValue"]);
const emit = defineEmits(["update:modelValue"]);
</script>

<template>
  <input
    :value="props.modelValue"
    @input="emit('update:modelValue', $event.target.value)"
  />
</template>
```
