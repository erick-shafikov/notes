компонент в кебаб-кейсе

у родителя в скрипте

```vue
<template>
  <some-component :propsName="somePropValye" @methodProps="someMethod" />
</template>

<script>
import CustomComponent from "./components/CustomComponent";
export default {
  components: { CustomComponent },
};
</script>
```

сам компонент

```vue
<!-- CustomComponent -->
<template>
  <div>{{ propsName }}</div>
</template>

<script>
export default {
  props: {
    propsName: {
      type: Boolean, //тип данных (класс JS)
      required: true, //обязательность
      default: someDefaultValue, //значение по умолчанию
      //или функция
      default() {
        return {
          some: "other",
        };
      },
      //валидация параметра
      validator(val) {
        return /^[1-9][1-9]+$/.test(val);
      },
      // не прописываются в пропсах
      methodProps,
    },
  },
  emits: {
    // если есть обработчики события с именем нативного события, можно предотвратить проброс нативного события
    input: null,
  },
  methods: {
    someEmitMethod() {
      this.$emit(
        "methodProps" /* здесь могут быть любые параметры, они будут доступны в $event параметре */
      );
    },
  },
};
</script>

<!-- стили -->
<!-- модификатор module инкапсулирует стили внутри компонента -->
<style module>
.appbtn {
  display: block !important;
  width: 100%;
}
</style>
```
