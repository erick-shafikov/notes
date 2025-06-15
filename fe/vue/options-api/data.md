Задает данные которые будет изменятся в последующем

```html
<input type="text" v-model="name" />
<hr />
<h2>Hello, {{ name }}</h2>

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
