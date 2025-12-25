# пользовательские директивы

Для создания v-подобных директив, для доступа к DOM

добавление в самом компоненте

```vue
<script setup>
// enables v-highlight in templates
const vHighlight = {
  mounted: (el) => {
    el.classList.add("is-highlight");
  },
};
</script>

<template>
  <p v-highlight>This sentence is important!</p>
</template>
```

На все случае жизненного цикла

```ts
const myDirective = {
  // called before bound element's attributes
  // or event listeners are applied
  created(el, binding, vnode) {
    // see below for details on arguments
  },
  // called right before the element is inserted into the DOM.
  beforeMount(el, binding, vnode) {},
  // called when the bound element's parent component
  // and all its children are mounted.
  mounted(el, binding, vnode) {},
  // called before the parent component is updated
  beforeUpdate(el, binding, vnode, prevVnode) {},
  // called after the parent component and
  // all of its children have updated
  updated(el, binding, vnode, prevVnode) {},
  // called before the parent component is unmounted
  beforeUnmount(el, binding, vnode) {},
  // called when the parent component is unmounted
  unmounted(el, binding, vnode) {},
};
```

Добавление в app

```ts
const app = createApp({});

// make v-highlight usable in all components
app.directive("highlight", {
  /* ... */
});
```

добавление в oapi

```ts
export default {
  setup() {
    /*...*/
  },
  directives: {
    // enables v-highlight in template
    highlight: {
      /* ... */
    },
  },
};
```
