# классы

## объектный синтаксис классов

```vue
<div :class="{ active: isActive }"></div>
<!-- если isActive===true -->
<div class="active"></div>
```

передача объекта в класс

```vue
<div :class="classObject"></div>
```

```js
const classObject = reactive({
  active: true,
  "text-danger": false,
});
```

## синтаксис с массивом

```html
<div :class="[activeClass, errorClass]"></div>
<div :class="[{ [activeClass]: isActive }, errorClass]"></div>
```

```js
const activeClass = ref("active");
const errorClass = ref("text-danger");
```

## использование атрибута стиля с массивом

```html
<!-- в классе не будет конфликта, будет конкатенация стилей-->
<div class="alert" :class="alertClasses">Some text</div>

<div v-for="val,key in tableClasses">
  <table class="table" :class="tableClasses"></table>
</div>

<!-- в стилях -->
<div class="alert alert-success transition-mh" :style="alertStyles"></div>
```

```js
let app = Vue.createApp({
  data: () => ({
    alertType: "green",
    tableClasses: {
      "table-hover": false,
      "table-bordered": false,
      "table-some": false,
    },
  }),
  computed: {
    alertClasses() {
      // возвращать объект вида класс: boolean
      return {
        "alert-success": this.alertType == "green",
        "alert-warning": this.alertType == "orange",
        "alert-danger": this.alertType == "red",
      };
    },
    computed: {
      alertStyles() {
        return {
          // возвращать стиль
          "min-height": this.alertHeight + "px",
        };
      },
    },
  },
});
app.mount(".sample");
```

# инлайн стили

```html
<div :style="{ color: activeColor, fontSize: fontSize + 'px' }"></div>
<div :style="{ 'font-size': fontSize + 'px' }"></div>
<div :style="styleObject"></div>
<!-- для нескольких -->
<div :style="[baseStyles, overridingStyles]"></div>
<!-- множественные значения -->
<div :style="{ display: ['-webkit-box', '-ms-flexbox', 'flex'] }"></div>
```

```js
const styleObject = reactive({
  color: "red",
  fontSize: "30px",
});
```
