# с массивами

- Vue может определять, когда вызываются методы мутации реактивного массива
- при немутирующих операциях
  ```js
  this.items = this.items.filter((item) => item.message.match(/Foo/));
  ```

работа с массивами

```js
data() {
  return {
    items: [{ message: 'Foo' }, { message: 'Bar' }]
  }
}
```

```vue
<!-- можно с of -->
<div v-for="item of items"></div>
<li v-for="item in items">{{ item.message }}</li>
<!-- с индексом -->
<li v-for="(item, index) in items">
  {{ parentMessage }} - {{ index }} - {{ item.message }}
</li>

<!-- вложенные поля -->
<li v-for="{ message } in items">{{ message }}</li>
<!--вложенные с индексом -->
<li v-for="({ message }, index) in items">{{ message }} {{ index }}</li>

<!-- вложенные списки -->
<li v-for="item in items">
  <span v-for="childItem in item.children">
    {{ item.message }} {{ childItem }}
  </span>
</li>
```

с объектами

```js
data() {
  return {
    myObject: {
      title: 'How to do lists in Vue',
      author: 'Jane Doe',
      publishedAt: '2016-04-10'
    }
  }
}
```

```vue
<!-- только значения -->
<li v-for="value in myObject">{{ value }}</li>
<!-- c ключами -->
<li v-for="(value, key) in myObject">{{ key }}: {{ value }}</li>
<!-- с индексом -->
<li v-for="(value, key, index) in myObject">
  {{ index }}. {{ key }}: {{ value }}
</li>
```

c числом

```vue
<span v-for="n in 10">{{ n }}</span>
```

v-for с v-if

```vue
<!--
v-if имеет более высокий приоритет, чем v-for. Это означает, что v-if условие не будет иметь доступа к переменным из области действия v-for
-->
<li v-for="todo in todos" v-if="!todo.isComplete">{{ todo.name }}</li>
<!-- исправить с помощью template -->
<template v-for="todo in todos">
  <li v-if="!todo.isComplete">{{ todo.name }}</li>
</template>
```

атрибут key может помочь при рендеринге

```vue
<div v-for="item in items" :key="item.id">
  <!-- content -->
</div>
<!-- использование с template обязательно -->
<template v-for="todo in todos" :key="todo.name">
  <li>{{ todo.name }}</li>
</template>
```

с компонентом

```jsx
<Component
  v-for="(item, index) in items"
  :item="item"
  :index="index"
  :key="item.id"
/>
```

# BPS:

## BP.Пример с добавлением элементов

```html
<button type="button" class="btn btn-primary" @click="add">Add number</button>

<ul class="list-group">
  добавляется на элемент списка
  <!-- num - элемент массива -->
  <!-- i - индекс массива -->
  <!-- numbers - массив из data -->
  <!-- при hover умножать на 2 -->
  <li v-for="num,i in numbers" class="list-group-item" @mouseenter="double(i)">
    <!-- можно достать элемент -->
    #{{ i }} / {{ num }}
  </li>
</ul>
```

```js
let app = Vue.createApp({
  data() {
    return {
      numbers: [],
    };
  },
  methods: {
    add() {
      this.numbers.push(Math.random());
    },
    double(i) {
      // при hover умножать на 2
      this.numbers[i] *= 2;
    },
  },
});
```
