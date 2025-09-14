# BPs

# BP. Масштабирование всего проекта c помощью font-size

```css
html {
  /* 10px/16px = 0.625 - теперь 1 rem 10px */
  font-size: 62.5%;
}

/* базовый шрифт – 16px, для удобства верстки удобно отталкиваться от 10px, моно сделать с помощью rem – условная единица от базового шрифта */

body {
  /* 30 px – задаем все размеры в rem */
  padding: 3rem;
}
```

# BP. иконка в конец ссылки

```scss
a[href*="http"] {
  // иконка будет отодвинута в правый край
  background: url("external-link-52.png") no-repeat 100% 0;
  background-size: 16px 16px;
  // padding отодвинет иконку от последней буквы
  padding-right: 19px;
}
```

# BP. ссылки кнопки

```scss
a {
  // отменяем все стилизации
  outline: none;
  text-decoration: none;
  // меняем блочность
  display: inline-block;
  // в примере 5 кнопок
  width: 19.5%;
  // с учетом того, что в примере 5 кнопок
  margin-right: 0.625%;
  text-align: center;
  line-height: 3;
  color: black;
}

a:link,
a:visited,
a:focus {
  background: yellow;
}

a:hover {
  background: orange;
}

a:active {
  background: red;
  color: white;
}
```

# BP. Текст, который залит фоном:

обрезка фона под текст

```css
.text-clip {
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
```

# BP. Улучшения по работе со шрифтами:

- использовать woff2
- пред загрузка
- фрагментация
- использование font-display
- локальное хранение шрифта

https://fonts.google.com/ - для поиска шрифтов

# размер шрифта под контейнер

```html
<span class="text-fit">
  <span>
    <span class="text-fit">
      <span><span>fit-to-width text</span></span>
      <span aria-hidden="true">fit-to-width text</span>
    </span>
  </span>
  <span aria-hidden="true">fit-to-width text</span>
</span>
```

```scss
.text-fit {
  display: flex;
  container-type: inline-size;

  --captured-length: initial;
  --support-sentinel: var(--captured-length, 9999px);

  & > [aria-hidden] {
    visibility: hidden;
  }

  & > :not([aria-hidden]) {
    flex-grow: 1;
    container-type: inline-size;

    --captured-length: 100cqi;
    --available-space: var(--captured-length);

    & > * {
      --support-sentinel: inherit;
      --captured-length: 100cqi;
      --ratio: tan(
        atan2(
          var(--available-space),
          var(--available-space) - var(--captured-length)
        )
      );
      --font-size: clamp(
        1em,
        1em * var(--ratio),
        var(--max-font-size, infinity * 1px) - var(--support-sentinel)
      );
      inline-size: var(--available-space);

      &:not(.text-fit) {
        display: block;
        font-size: var(--font-size);

        @container (inline-size > 0) {
          white-space: nowrap;
        }
      }

      /* Necessary for variable fonts that use optical sizing */
      &.text-fit {
        --captured-length2: var(--font-size);
        font-variation-settings: "opsz" tan(atan2(var(--captured-length2), 1px));
      }
    }
  }
}

@property --captured-length {
  syntax: "<length>";
  initial-value: 0px;
  inherits: true;
}

@property --captured-length2 {
  syntax: "<length>";
  initial-value: 0px;
  inherits: true;
}
```

альтернатива через svg

```html
<h1 class="container">
  <svg>
    <text>Fit text to container</text>
  </svg>
</h1>
```

```scss
h1.container {
  /* Container size */
  width: 100%;

  /* Type styles (<text> will inherit most of them) */
  font: 900 1em system-ui;
  color: hsl(43 74% 3%);

  text {
    /*
      We have to use fill: instead of color: here
      But we can use currentColor to inherit the color
    */
    fill: currentColor;
  }
}
```

```js
/* Select all SVGs */
const svg = document.querySelectorAll("svg");

/* Loop all SVGs */
svg.forEach((element) => {
  /* Get bounding box of <text> element */
  const bbox = element.querySelector("text").getBBox();
  /* Apply bounding box values to SVG element as viewBox */
  element.setAttribute(
    "viewBox",
    [bbox.x, bbox.y, bbox.width, bbox.height].join(" ")
  );
});
```
