<!-- attr() ---------------------------------------------------------------------------------------------------------------------------->

# attr()

Можно доставать значение элемента

```scss
attr(data-count);
attr(title);

/* С типом */
attr(src url);
attr(data-count number);
attr(data-width px);

/* с фоллбэком */
attr(data-count number, 0);
attr(src url, '');
attr(data-width px, inherit);
attr(data-something, 'default');
```

Пример использования

Добавит слово hello в качестве ::before, достанет из аттрибута

```html
<p data-foo="hello">world</p>
```

```scss
p::before {
  content: attr(data-foo) " ";
}
```

Использование с атрибутами - свойствами

```html
<body>
  <div class="awesome-block" data-rotation="45deg"></div>
</body>
```

```scss
.awesome-block {
  width: 200px;
  height: 200px;
  background-color: purple;
  // type поможет распознать, что rotation - angle
  rotate: attr(data-rotation type(<angle>));
}
```

```html
<div class="awesome-block" data-size="450px"></div>
<div class="awesome-block" data-size="10rem"></div>
```

```scss
.awesome-block {
  width: attr(data-size type(<length-percentage>));
  height: attr(data-size type(<length-percentage>));
  background-color: purple;
}
```

```html
<div class="awesome-block" data-size="450"></div>
<div class="awesome-block" data-size="10"></div>
```

```scss
.awesome-block {
  width: attr(data-size px);
  height: attr(data-size px);
  background-color: purple;
}
```

резервные значения

```scss
.awesome-block {
  width: attr(data-size type(<length-percentage>), 50px);
  height: attr(data-size type(<length-percentage>), 50px);
  background-color: purple;
}
```

<!-- env()---------------------------------------------------------------------------------------------------------------------------------->

# env()

Позволяет получить значение какого-либо свойства предопределенное системой

```scss
body {
  padding: env(safe-area-inset-top, 20px) env(safe-area-inset-right, 20px) env(
      safe-area-inset-bottom,
      20px
    ) env(safe-area-inset-left, 20px);
}
```

Значения:

- safe-area-inset-top, safe-area-inset-right, safe-area-inset-bottom, safe-area-inset-left
- titlebar-area-x, titlebar-area-y, titlebar-area-width, titlebar-area-height
- keyboard-inset-top, keyboard-inset-right, keyboard-inset-bottom, keyboard-inset-left, keyboard-inset-width, keyboard-inset-height

# if

```scss
body {
  // media(), style(), supports()
  background-color: if(media(width < 600px): tomato; else: purple);
}
```

c переменными

```scss
body {
  background-color: if(style(--color: white): tomato; else: purple;);
  --color: black;
}

@media (width < 1000px) {
  body {
    --color: white;
  }
}
```

<!-- is() ---------------------------------------------------------------------------------------------------------------------------------->

# is()

позволяет проверить на наличие поддержки того или иного свойства

```scss
is(--moz-prefix) {
  //
}
```

<!-- layer() ---------------------------------------------------------------------------------------------------------------------------->

# layer()

для работы с пространствами

```scss
 {
  @import url layer(layer-name);
  @import "dark.css" layer(framework.themes.dark);
}
```

<!-- light-dark() ---------------------------------------------------------------------------------------------------------------------------->

# light-dark()

для определения темы

```scss
 {
  :root {
    color-scheme: light dark;
  }
  body {
    color: light-dark(#333b3c, #efefec);
    background-color: light-dark(#efedea, #223a2c);
  }
}
```

```scss
:root {
  /* this has to be set to switch between light or dark */
  color-scheme: light dark;

  --light-bg: ghostwhite;
  --light-color: darkslategray;
  --light-code: tomato;

  --dark-bg: darkslategray;
  --dark-color: ghostwhite;
  --dark-code: gold;
}
* {
  background-color: light-dark(var(--light-bg), var(--dark-bg));
  color: light-dark(var(--light-color), var(--dark-color));
}
code {
  color: light-dark(var(--light-code), var(--dark-code));
}
```

<!-- paint() ---------------------------------------------------------------------------------------------------------------------------->

# paint() (-ff -s)

Декорация для PaintWorkletGlobalScope

```scss
li {
  background-image: paint(boxbg);
  --boxColor: hsl(55 90% 60% / 100%);
}
li:nth-of-type(3n) {
  --boxColor: hsl(155 90% 60% / 100%);
  --widthSubtractor: 20;
}
li:nth-of-type(3n + 1) {
  --boxColor: hsl(255 90% 60% / 100%);
  --widthSubtractor: 40;
}
```

<!-- url()---------------------------------------------------------------------------------------------------------------------------->

# url()

использование внешних ресурсов

```scss
 {
  background-image: url("star.gif");
  list-style-image: url("../images/bullet.jpg");
  content: url("my-icon.jpg");
  cursor: url(my-cursor.cur);
  border-image-source: url("/media/diamonds.png");
  src: url("fantastic-font.woff");
  offset-path: url(#path);
  mask-image: url("masks.svg#mask1");

  /* Properties with fallbacks */
  cursor: url(pointer.cur), pointer;

  /* Associated short-hand properties */
  background: url("star.gif") bottom right repeat-x blue;
  border-image: url("/media/diamonds.png") 30 fill / 30px / 30px space;

  /* As a parameter in another CSS function */
  background-image: cross-fade(20% url(first.png), url(second.png));
  mask-image: image(
    url(mask.png),
    skyblue,
    linear-gradient(rgb(0 0 0 / 100%), transparent)
  );

  /* as part of a non-shorthand multiple value */
  content: url(star.svg) url(star.svg) url(star.svg) url(star.svg) url(star.svg);

  /* at-rules */
  @document url(https://www.example.com/)
  {
    /* … */
  }
  @import url("https://www.example.com/style.css");
  @namespace url(http://www.w3.org/1999/xhtml);
}
```

<!-- var() ---------------------------------------------------------------------------------------------------------------------------->

# var()

```scss
 {
  .component .header {
    color: var(
      --header-color,
      blue
    ); /* header-color не существует, поэтому используется blue */
  }
}
```
