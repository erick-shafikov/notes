Имя name будет доступно в шаблоне

```html
<h2>Hello, {{ name }}</h2>

<script>
  let app = Vue.createApp({
    data() {
      return {
        name: "",
      };
    },
  });

  let root = app.mount(".sample");
</script>
```
