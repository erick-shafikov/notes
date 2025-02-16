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

Модификаторы:

- .stop - вызывает event.stopPropagation().
- .prevent - вызывает event.preventDefault().
- .capture - добавить обработчик событий в capture режиме.
- .self - запускать обработчик только в том случае, если событие было отправлено именно от этого элемента.
- .{keyAlias} - запускать обработчик только по определенным клавишам.
- .once - обработчик сработает только один раз.
- .left - обработчик срабатывания только для событий левой кнопки мыши.
- .right - обработчик срабатывания только для событий правой кнопки мыши.
- .middle - обработчик срабатывания только для событий средней кнопки мыши.
- .passive - добавляет обработчик DOM события с параметром { passive: true }.

```html
<!-- метод в качестве обработчика  -->
<button v-on:click="doThis"></button>
<!-- динамическое событие -->
<button v-on:[event]="doThis"></button>
<!-- inline-выражение -->
<button v-on:click="doThat('hello', $event)"></button>
<!-- сокращённая запись -->
<button @click="doThis"></button>
<!-- окращённая запись динамического события -->
<button @[event]="doThis"></button>
<!-- stop propagation -->
<button @click.stop="doThis"></button>
<!-- prevent default -->
<button @click.prevent="doThis"></button>
<!-- prevent default без выражения -->
<form @submit.prevent></form>
<!-- цепочка из модификаторов -->
<button @click.stop.prevent="doThis"></button>
<!-- модификатор клавиши с использованием keyAlias -->
<input @keyup.enter="onEnter" />
<!-- обработчик события будет вызван не больше одного раза -->
<button v-on:click.once="doThis"></button>
<!-- объектный синтаксис -->
<button v-on="{ mousedown: doThis, mouseup: doThat }"></button>
```
