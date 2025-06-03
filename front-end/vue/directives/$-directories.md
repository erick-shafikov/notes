<!-- $ref --------------------------------------------------------------->

# $ref

Позволяет получить ссылку на DOM элемент

```html
<input ref="firstInput" />
```

```js
Vue.createApp({
  mounted() {
    // обращение к ref
    this.$refs.firstInp.focus();
  },
});
```

в цикле

```html
<div class="form-group" v-for="guest,i in guests">
  <input
    v-model.trim="guest.value"
    type="text"
    class="form-control"
    ref="guests"
  />
</div>
```

```js
Vue.createApp({
  addGuest() {
    this.guests.push({ value: "" });

    this.$nextTick(() => {
      let guests = this.$refs.guests;
      guests[guests.length - 1].focus();
    });
  },
});
```
