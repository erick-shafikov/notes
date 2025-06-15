основная цель - вычисления на основе полей data и присвоение вычисленных значений полям из data. Методы не вызываются из шаблона

Преимущества:

- Методы предоставляют понятный и явный способ определения поведения внутри компонентов Vue.
- Их можно использовать повторно и вызывать из разных мест внутри компонента.
- Методы могут принимать параметры, обеспечивая динамическое поведение на основе входных данных.

Использование:

- Обработка событий: методы обычно используются для обработки взаимодействий пользователя, таких как нажатия кнопок или отправка форм.
- Преобразования данных: Методы подходят для выполнения сложных преобразований данных или вычислений.
- Асинхронные операции: методы могут обрабатывать асинхронные задачи, такие как извлечение данных из API или выполнение сетевых запросов.
- Используйте методы, когда вам необходимо выполнить императивные действия или сложную логику, не зависящую от реактивных изменений данных.

```html
<!-- связь с методом onInput в methods -->
<input type="text" :value="name" @input="onInput" />

<script>
  let app = Vue.createApp({
    data() {
      return {
        name: "",
      };
    },
    methods: {
      // могут быть обработчиками событий
      //будет передан в качестве @input
      onInput(e) {
        this.name = e.target.value;
      },
    },
  });

  let root = app.mount(".sample");
</script>
```

```html
<!-- не рекомендуется, так как methods для работы с данными -->
<input type="text" v-model.trim="firstName" />
<input type="text" v-model.trim="lastName" />
{{ toUpper(firstName) }} {{ toUpper(lastName) }}

<script>
  let app = Vue.createApp({
    data() {
      return {
        showAlert: false,
        firstName: "",
        lastName: "",
      };
    },
    computed: {
      fullName() {
        console.log("here");
        return (this.firstName + " " + this.lastName).trim();
      },
    },
    methods: {
      toUpper(str) {
        return str.toUpperCase().split("").reverse().join("");
      },
    },
  });

  let root = app.mount(".sample");
</script>
```

# асинхронные вызовы

```html
<input type="text" v-model.trim.lazy="promo" @change="checkSale" /><br />
{{ price }}
<div v-if="hasSale">
  <div class="alert alert-danger">- {{ sale }}%</div>
  {{ total }}
</div>

<script>
  let app = Vue.createApp({
    data() {
      return {
        promo: "",
        price: 1000,
        sale: 0,
      };
    },
    computed: {},
    methods: {
      // вспомогательный метод для вызова асинхронного
      checkSale() {
        this.getSale(this.promo, (sale) => {
          this.sale = sale;
        });
      },
      //асинхронный метод
      getSale(promo, fn) {
        setTimeout(function () {
          let codes = {
            some: 10,
            other: 20,
          };

          let sale = codes.hasOwnProperty(promo) ? codes[promo] : 0;
          fn(sale);
        }, 500);
      },
    },
  });

  let root = app.mount(".sample");
</script>
```
