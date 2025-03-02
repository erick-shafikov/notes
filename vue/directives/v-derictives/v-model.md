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

```html
<input v-model="name" />

<input v-bind:value="name" v-on:input="name = $event.target.value" />
```

в данном приме

```html
<div class="wrapper">
  <div class="sample">
    <input type="text" v-model="name" />
    <hr />
    <h2>Hello, {{ name }}</h2>
  </div>
</div>

<script>
  let app = Vue.createApp({
    data() {
      return {
        name: "",
      };
    },
  });
</script>
```

# c чекбоксами

```vue
<template>
  <input type="checkbox" id="checkbox" v-model="checked" />
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

# c радо кнопками

```vue
<template>
  <div>Выбрано: {{ picked }}</div>

  <input type="radio" id="one" value="Один" v-model="picked" />
  <label for="one">Один</label>

  <input type="radio" id="two" value="Два" v-model="picked" />
  <label for="two">Два</label>
</template>
```

# c селектом

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
