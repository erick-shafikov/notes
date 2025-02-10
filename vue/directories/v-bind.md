v-bind

тоже самое что и :value="some"

```html
<input
  type="text"
  v-bind:value="name"
  v-on:input="name = $event.target.value"
/>

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
