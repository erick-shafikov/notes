```vue
<template>
  <app-alert title="Hello, user">
    <!-- анонимный слот -->
    <p>Total: {{ total }}</p>
    <p>Items: {{ items }}</p>
    <!-- именованный слот с компонентом по умолчанию -->
    <template v-slot:footer>
      <button type="button" class="btn btn-danger mr-1" @click="reset">
        Reset
      </button>
      <button type="button" class="btn btn-primary">Send</button>
    </template>
  </app-alert>
</template>
<script>
export default {};
</script>
```

компонент со слотом

```vue
<template>
  <div class="alert" :class="classes">
    <h4>{{ title }}</h4>
    <hr />
    <div>
      <slot></slot>
    </div>
    <hr />
    <slot name="footer">
      <button type="button" class="btn btn-primary" @click="$emit('ok')">
        Ok
      </button>
    </slot>
  </div>
</template>
<script>
export default {
  props: {
    title: String,
    color: { type: String, default: "success" },
  },
  computed: {
    classes() {
      return "alert-" + this.color;
    },
  },
};
</script>
```
