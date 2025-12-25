# scoped

тег scoped добавит data-v-123123 атрибут, которы обеспечит сокрытие этой кнопки, так как ее стили превратятся в button[ata-v=123123]. Нужно для изоляции. С этим тегом стили не распространяются на дочерние компоненты

```vue
<script setup></script>

<template>
  <button class="button">Кнопка</button>
</template>

<style scoped>
.button {
  border: none;
  border-radius: 10px;
}
</style>
```

## вложенные классы

для того что бы распространялся. Если определить класс b выше и вложить его в компонент

```vue
<style scoped>
.a :deep(.b) {
  /* ... */
}
</style>
```

## для слотов

```vue
<style scoped>
:slotted(div) {
  color: red;
}
</style>
```

# глобальные

```vue
<style scoped>
:global(.red) {
  color: red;
}
</style>
```

# объектный синтаксис классов

```vue
<div :class="{ active: isActive }"></div>
<!-- если isActive===true -->
<div class="active"></div>
```

передача объекта в класс

```vue
<script>
const classObject = reactive({ active: true, "text-danger": false });
</script>

<template>
  <div :class="classObject"></div>
</template>
```

# синтаксис с массивом

```vue
<script>
const activeClass = ref("active");
const errorClass = ref("text-danger");
</script>

<template>
  <div :class="[activeClass, errorClass]"></div>
  <div :class="[{ [activeClass]: isActive }, errorClass]"></div>
</template>
```

## использование атрибута стиля с массивом

```vue
<script>
const alertType = "green";

const tableClasses = {
  "table-hover": false,
  "table-bordered": false,
  "table-some": false,
};

const alertClasses = {
  "alert-success": alertType == "green",
  "alert-warning": alertType == "orange",
  "alert-danger": alertType == "red",
};

const alertStyles = {
  "min-height": this.alertHeight + "px",
};
</script>

<template>
  <!-- в классе не будет конфликта, будет конкатенация стилей-->
  <div class="alert" :class="alertClasses">Some text</div>

  <div v-for="(val, key) in tableClasses">
    <table class="table" :class="tableClasses"></table>
  </div>

  <!-- в стилях -->
  <div class="alert alert-success transition-mh" :style="alertStyles"></div>
</template>
```

# инлайн стили

```vue
<script>
const styleObject = reactive({
  color: "red",
  fontSize: "30px",
});
</script>

<template>
  <div :style="{ color: activeColor, fontSize: fontSize + 'px' }"></div>
  <div :style="{ 'font-size': fontSize + 'px' }"></div>
  <div :style="styleObject"></div>
  <!-- для нескольких -->
  <div :style="[baseStyles, overridingStyles]"></div>
  <!-- множественные значения -->
  <div :style="{ display: ['-webkit-box', '-ms-flexbox', 'flex'] }"></div>
</template>
```

# c компонентами

```vue
<!-- шаблон дочернего компонента -->
<p class="foo bar">Привет</p>
<!-- при использовании компонента -->
<MyComponent class="baz boo" />
<!-- результат -->
<p class="foo bar baz boo">Привет</p>
```

# css модули

style module будут генерить объекты, которы будут подставлены в :class их можно использовать в качестве объекта

```vue
<template>
  <p :class="$style.red">This should be red</p>
</template>

<style module>
.red {
  color: red;
}
</style>
```

если нужно избавить от $style

```vue
<template>
  <p :class="classes.red">red</p>
</template>

<style module="classes">
.red {
  color: red;
}
</style>
```

можно получить с помощью хука

```vue
<script setup lang="ts">
import { useCssModule } from "vue";

const classes = useCssModule();
</script>

<template>
  <p :class="classes.red">red</p>
</template>

<style module>
.red {
  color: red;
}
</style>
```

# v-bind

```vue
<script setup>
import { ref } from "vue";
const theme = ref({
  color: "red",
});
</script>

<template>
  <p>hello</p>
</template>

<style scoped>
p {
  color: v-bind("theme.color");
}
</style>
```
