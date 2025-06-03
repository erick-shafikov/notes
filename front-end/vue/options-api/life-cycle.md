# жизненный цикл

```js
let app = Vue.createApp({
  template: `
				<div class="form-group">
					<label>Email</label>
					<input type="text" class="form-control" v-model="email"><hr>
					{{ email }}
				</div>
			`,
  data: () => ({
    email: "test email",
  }),
  beforeCreate() {
    console.log("beforeCreate");
    console.log(this.email); //undefined
  },
  created() {
    console.log("created");
    console.log(this.email); //test email
    console.log(this.$el); //null
  },
  beforeMount() {
    console.log("beforeMounted");
    console.log(this.$el); //null
  },
  mounted() {
    console.log("mounted");
    console.log(this.$el); //input
    console.log(this.$el.innerHTML); //внутренности
  },
  beforeUpdate() {
    console.log("beforeUpdate"); //при вводе
  },
  updated() {
    console.log("updated"); //после ввода
  },
});

app.mount(".sample");
```

# mounted

метод срабатывает, когда компонент вмонтирован в дерево DOM

обращение к ссылкам на элементы

```html
<input ref="firstInput" />
```

```js
Vue.createApp({
  methods() {
    this.$refs.firstInput;
  },
  mounted() {
    // обращение к ref
    this.$refs.firstInp.focus();
  },
});
```
