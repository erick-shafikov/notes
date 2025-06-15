nextTick будет запущен после ре-рендеринга шаблона

```html
<input ref="firstInput" />
```

```js

methods(){
  this.$nextTick(() => {this.$refs.firstInput.input()})

}

```
