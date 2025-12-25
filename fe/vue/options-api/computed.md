# computed

основная цель - производить операции с данными в data и выводить в шаблон

Преимущества:

- реактивный
- понятный и краткий способ получения значений из существующих данных, повышают читаемость
- Они автоматически кэшируют свои результаты

Использование:

- Производные данные
- Условная визуализация
- Кэширование дорогостоящих операций
- необходимо извлечь значения из реактивных данных и вы хотите воспользоваться преимуществами системы реактивности и механизма кэширования Vue.

```html
<div class="sample">
  <!-- будет изменяться firstName -->
  FN: <input type="text" v-model.trim="firstName" /><br />
  <!-- будет изменяться lastName -->
  LN: <input type="text" v-model.trim="lastName" /><br />
  <hr />
  <!-- результат computed поля fullName -->
  <h2>Hello, {{ fullName }}</h2>
</div>
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
        return (this.firstName + " " + this.lastName).trim();
      },
    },
  });

  let root = app.mount(".sample");
</script>
```

альтернативный вариант воспользоваться watch

# взаимодействие с methods

computed может рассчитывать на основе methods

```html
<input type="text" v-model.trim.lazy="promo" /><br />

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
      };
    },
    computed: {
      sale() {
        // вызова getSale из methods так как на основе денных, синхронно
        return this.getSale(this.promo);
      },
      hasSale() {
        return this.sale > 0;
      },
      total() {
        return this.price * (1 - this.sale / 100);
      },
    },
    methods: {
      getSale(promo) {
        let codes = {
          some: 10,
          other: 20,
        };

        return codes.hasOwnProperty(promo) ? codes[promo] : 0;
      },
    },
  });

  let root = app.mount(".sample");
</script>
```
