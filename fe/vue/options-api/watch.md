Позволяет следить за полями data

```html
<input type="text" v-model.trim.lazy="promo" />
<button type="button" @click="promo = ''">X</button>

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
    computed: {
      hasSale() {
        return this.sale > 0;
      },
      total() {
        return this.price * (1 - this.sale / 100);
      },
    },
    watch: {
      // поле будет следить за promo и запустит getSale из methods
      promo() {
        this.getSale(this.promo, (sale) => {
          this.sale = sale;
        });
      },
    },
    methods: {
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
