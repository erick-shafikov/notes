v-on

c модификаторами v-on:click.prevent.once @click=""

аналогично

```js
input.addEventListener("input", function (e) {
  let $event = e;
  this.name = $event.target.value;
});
```

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
