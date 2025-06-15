```html
<app-bool-to-text :value="true" trueclass="text-primary" truetext="Да" />
```

```js
Vue.createApp({
  components: {
    AppBoolToText,
  },
}).mount(".sample");

app.component("app-some", {
  props: {
    value: {
      type: Boolean,
      required: true,
      default() {
        return {
          some: "other",
        };
      },
      validator(val) {
        return /^[1-9][1-9]+$/.test(val);
      },
    },
  },
  computed: {
    classes() {
      return this.value ? this.trueclass : this.falseclass;
    },
    text() {
      return this.value ? this.truetext : this.falsetext;
    },
  },
  template: `<span :class="classes">{{ text }}</span>`,
});
```

```html
<app-todo-action
  v-for="action,i in todoList"
  :key="i"
  :title="action.title"
  :value="action.current"
  :max="action.max"
  @step="makeStep(i)"
></app-todo-action>
```

```js
let AppTodoAction = {
  props: {
    title: { type: String, required: true },
    value: { type: Number, required: true },
    max: { type: Number, required: true },
  },
  computed: {
    rel() {
      return this.value / this.max;
    },
    progressStyles() {
      return { width: this.rel * 100 + "%" };
    },
    alertClasses() {
      return {
        "alert-danger": this.rel < 0.25,
        "alert-warning": this.rel >= 0.25 && this.rel < 0.75,
        "alert-success": this.rel >= 0.75,
      };
    },
  },
  methods: {
    step() {
      this.$emit("step" /* , { some: 'nz' } */);
    },
  },
  template: `
    <div class="action">
			<div class="alert" :class="alertClasses">
				<h2>{{ title }}</h2>
				<div class="progress">
					<div class="progress-bar" :style="progressStyles"></div>
				</div>
				<hr>
				<h3 v-if="value == max">All done!</h3>
				<button v-else @click="step" type="button" class="btn btn-primary">
					I make step!
				</button>
			</div>
		</div>
  `,
};
```
