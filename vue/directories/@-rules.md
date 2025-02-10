# @click

```html
<a v-on:click="doSomething"> ... </a>

<!-- shorthand -->
<a @click="doSomething"> ... </a>
```

```html
<!-- add из methods -->
<button
  type="button"
  class="btn 
btn-primary"
  @click="add"
>
  Add number
</button>

<script>
  let app = Vue.createApp({
    data() {},
    methods: {
      add() {},
    },
  });

  let root = app.mount(".sample");
</script>
```
