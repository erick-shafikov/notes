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

для клавиш:
.enter, .tab, .delete (ловит как «Delete», так и «Backspace»), .esc, .space, .up, .down, .left, .right, .ctrl, .alt, .shift, .meta

.exact - для точного совпадения

```vue
<template>
  <input @keyup.page-down="onPageDown" />
  <!-- Alt + Enter -->
  <input @keyup.alt.enter="clear" />

  <!-- Ctrl + Click -->
  <div @click.ctrl="doSomething">Сделать что-нибудь</div>
  <!-- сработает, даже если также будут нажаты Alt или Shift -->
  <button @click.ctrl="onClick">A</button>

  <!-- сработает, только когда нажат Ctrl и не нажаты никакие другие клавиши -->
  <button @click.ctrl.exact="onCtrlClick">A</button>

  <!-- сработает, только когда не нажаты никакие системные модификаторы -->
  <button @click.exact="onClick">A</button>
</template>
```

# варианты обработчиков события

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

# Инлайн обработка

<!--  -->

```vue
<template>
  <button @click="count++">Добавить 1</button>
  <p>Счётчик: {{ count }}</p>
</template>

<script>
const count = ref(0);
</script>
```

# метод

```vue
<template>
  <!-- `greet` — название метода, объявленного в компоненте выше -->
  <button @click="greet">Поприветствовать</button>
</template>

<script>
const name = ref("Vue.js");

function greet(event) {
  alert(`Привет, ${name.value}!`);
  // `event` — нативное событие DOM
  if (event) {
    alert(event.target.tagName);
  }
}
</script>
```

# передача аргументов

```vue
<template>
  <button @click="say('привет')">Скажи привет</button>
  <button @click="say('пока')">Скажи пока</button>
</template>

<script>
function say(message) {
  alert(message);
}
</script>
```

<!--  -->

```vue
<template></template>

<script></script>
```
