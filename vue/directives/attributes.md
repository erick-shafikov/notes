```html
<button :disabled="shouldButtonDisable"></button>
<!--  Если связанное значение равно null или undefined, то атрибут будет удален из отображаемого элемента. -->
<div v-bind:id="dynamicId"></div>
<!-- краткая запись -->
<div :id="dynamicId"></div>
<!-- same as :id="id" -->
<div :id></div>
<div v-bind:id></div>
```

связать автоматически

```js
data() {
  return {
    objectOfAttrs: {
      id: 'container',
      class: 'wrapper'
    }
  }
}
```

```html
<!-- id: 'container', class: 'wrapper' -->
<div v-bind="objectOfAttrs"></div>
```

выражения

```html
<div :id="`list-${id}`"></div>
```

- window, не будут доступны в выражениях шаблонов.

динамические

```vue
<template>
<a v-bind:[attributeName]="url"> ... </a>
<!-- shorthand -->
<a :[attributeName]="url"> ... </a>

<a v-on:[eventName]="doSomething"> ... </a>
<!-- shorthand -->
<a @[eventName]="doSomething"> ... </a>


<!-- нельзя -->
<a :['foo' + bar]="value"> ... </a>
</template>
```
