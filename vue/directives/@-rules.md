<!-- @click ---------------------------------------------------------------------------------------------------------------------------------->

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

<!-- @submit --------------------------------------------------------------------------------------------------------------------------------->

# @submit

```html
<form v-if="!formDone" @submit.prevent="sendForm"></form>
```

```js
Vue.createApp({
  methods: {
    sendForm() {
      // код функции
    },
  },
  mounted() {
    this.$refs.firstInp.focus();
  },
}).mount(".sample");
```
