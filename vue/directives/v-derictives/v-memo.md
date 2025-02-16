Мемоизация части древа шаблона

```html
<div v-memo="[valueA, valueB]">...</div>
```

v-memo не работает внутри v-for

```html
<div v-for="item in list" :key="item.id" v-memo="[item.id === selected]">
  <p>ID: {{ item.id }} - выбран: {{ item.id === selected }}</p>
  <p>...больше дочерних элементов</p>
</div>
```
