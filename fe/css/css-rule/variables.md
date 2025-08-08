# Переменные

```scss
$base-color: #c6538c;

.alert {
  border: 1px solid $base-color;
}
```

inherit работает как и со всеми свойствами, в данном случае будет lightblue

```scss
:root {
  --page-background-color: lightblue;
  background-color: tomato;
}

body {
  --page-background-color: inherit;
  background-color: var(--page-background-color);
}
```

# BP. Псевдо миксин для размера шрифта

КонцепцияЖ

```scss
.atom-class {
  // если есть color, то --_color == --color если нет то #fff
  --_color: var(--color, #fff);

  color: var(--_color);
}

// контроллер
.my-text {
  --color: #f00;
}
```

```html
<!-- т.е. если есть my-text => #f00 -->
<span class="my-text atom-class">Hallo, world!!!</span>
```

Пример для размера шрифта

```scss
[class*="static-font"] {
  --_font-size: var(--font-size, 1em);
  --_line-height: var(--line-height, calc(var(--_font-size) + 4px));

  font-size: var(--_font-size);
  line-height: var(--_line-height);
}

.static-font__M {
  --font-size: 20px;
  --line-height: 26px;

  @media (max-width: 1024px) and (min-width: 510px) {
    --font-size: 18px;
    --line-height: 22px;
  }

  @media (max-width: 509px) {
    --font-size: 16px;
    --line-height: 20px;
  }
}
```

```scss
[class*="responsive-font"] {
  /* Require props */
  --_max-font-size: var(--max-font-size);
  --_max-line-height: var(--max-line-height);
  --_max-screen-width: var(--max-screen-width);

  --_min-font-size: var(--min-font-size);
  --_min-line-height: var(--min-line-height);
  --_min-screen-width: var(--min-screen-width);
  /* ============= */

  /* Computed deltas */
  --font-delta: (var(--_max-font-size) - var(--_min-font-size));
  --line-height-delta: (var(--_max-line-height) - var(--_min-line-height));
  --screen-width-delta: (var(--_max-screen-width) - var(--_min-screen-width));
  /* =============== */

  --main-coef: (100vw - var(--_min-screen-width) * 1px) / var(
      --screen-width-delta
    );

  /* Target values */
  --computed-font-size: calc(
    var(--_min-font-size) * 1px + var(--font-delta) * var(--main-coef)
  );
  --computed-line-height: calc(
    var(--_min-line-height) + var(--line-height-delta) * var(--main-coef)
  );
  /* ============= */

  font-size: clamp(
    calc(var(--_min-font-size) * 1px),
    var(--computed-font-size),
    calc(var(--_max-font-size) * 1px)
  );
  line-height: clamp(
    calc(var(--_min-line-height) * 1px),
    var(--computed-line-height),
    calc(var(--_max-font-size) * 1px)
  );
}

.responsive-font__M {
  --max-screen-width: var(--desktop-size-max);
  --max-font-size: 20;
  --max-line-height: 26;

  --min-screen-width: var(--tablet-size-max);
  --min-font-size: 18;
  --min-line-height: 22;

  @media (max-width: 1024px) and (min-width: 510px) {
    --max-screen-width: var(--tablet-size-max);
    --max-font-size: 18;
    --max-line-height: 22;

    --min-screen-width: var(--mobile-size-max);
    --min-font-size: 16;
    --min-line-height: 20;
  }

  @media (max-width: 509px) {
    --max-screen-width: var(--mobile-size-max);
    --max-font-size: 16;
    --max-line-height: 20;

    --min-screen-width: var(--mobile-size-min);
    --min-font-size: 16;
    --min-line-height: 20;
  }
}

:root {
  /* ===== Desktop ===== */

  --desktop-size-max: 1920;
  --desktop-size-min: 1441;

  /* =================== */

  /* ===== Laptop ===== */

  --laptop-size-max: 1440;
  --laptop-size-min: 1025;

  /* ================== */

  /* ===== Tablet ===== */

  --tablet-size-max: 1024;
  --tablet-size-min: 510;

  /* ================== */

  /* ===== Mobile ===== */

  --mobile-size-max: 509;
  --mobile-size-min: 350;

  /* ================== */
}
```
