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
