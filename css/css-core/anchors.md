<!-- Якоря ------------------------------------------------------------------------------------------------------------------------------->

# Якоря

Позволяют разместить один элемент относительно другого.

```html
<!-- якорь -->
<div class="anchor">⚓︎</div>

<!-- контент якоря -->
<div class="infoBox">
  <p>You can edit this text.</p>
</div>
```

```scss
.anchor {
  // задаем имя для элемента, около которого происходит позиционирование
  anchor-name: --myAnchor;
}

.infoBox {
  position-anchor: --myAnchor;
  // обязательно нужно что бы position === fixed или position === absolute
  position: fixed;
  opacity: 0.8;
  inset-area: top left;
}
```

# anchor-name (на якорь)

добавляет имя для якоря, к которому в последствии присоединится элемент

```scss
 {
  //
  anchor-name: --name;
  // для нескольких
  anchor-name: --name, --another-name;
}
```

# inset-block

```scss
 {
  inset-block-start: 3px | 1rem | anchor(end) | calc(
      anchor(--myAnchor 50%) + 5px
    )
    | 10%
    // расположит элемент якоря, аналогично inset-block-end ,inset-inline-start, inset-inline-end
;
  inset-block: 10px 20px; // определяет начальные и конечные смещения логического блока элемента, аналогично inset-inline
  inset: ; // inset-block-start + inset-block-end + inset-inline-start + inset-inline-end
}
```

# inset-area (где относительно якоря)

Нестабильное свойство. Позволяет позиционировать якорь

```scss
 {
  inset-area: none;

  //  для одного заголовка
  inset-area: top left;
  inset-area: start end;
  inset-area: block-start center;
  inset-area: inline-start block-end;
  inset-area: x-start y-end;
  inset-area: center y-self-end;
  // для двух
  inset-area: top span-left;
  inset-area: center span-start;
  inset-area: inline-start span-block-end;
  inset-area: y-start span-x-end;
  // для трех
  inset-area: top span-all;
  inset-area: block-end span-all;
  inset-area: x-self-start span-all;

  /* One <inset-area> keyword with an implicit second <inset-area> keyword  */
  inset-area: top; /* equiv: top span-all */
  inset-area: inline-start; /* equiv: inline-start span-all */
  inset-area: center; /* equiv: center center */
  inset-area: span-all; /* equiv: center center */
  inset-area: end; /* equiv: end end */

  /* Global values */
  inset-area: inherit;
  inset-area: initial;
  inset-area: revert;
  inset-area: revert-layer;
  inset-area: unset;
}
```

# position-anchor (относительно какого якоря)

Ограниченная доступность. Определяет имя якоря элемента. Актуально только для позиционированных элементов

```scss
 {
  position-anchor: --anchorName; //имя якоря относительного которого будет происходить позиционирование
  position: fixed | absolute;
}
```

# position-area (где относительно якоря)

размещение якоря, альтернатива для функции anchor()

```scss
.position-area {
  position-area: top left;
  position-area: start end;
  position-area: block-start center;
  position-area: inline-start block-end;
  position-area: x-start y-end;
  position-area: center y-self-end;

  /* Two <position-area> keywords spanning two tiles */
  position-area: top span-left;
  position-area: center span-start;
  position-area: inline-start span-block-end;
  position-area: y-start span-x-end;

  /* Two <position-area> keywords spanning three tiles */
  position-area: top span-all;
  position-area: block-end span-all;
  position-area: x-self-start span-all;

  /* One <position-area> keyword with an implicit second <position-area> keyword  */
  position-area: top; /* equiv: top span-all */
  position-area: inline-start; /* equiv: inline-start span-all */
  position-area: center; /* equiv: center center */
  position-area: span-all; /* equiv: center center */
  position-area: end; /* equiv: end end */
}
```

# position-try:

position-try-order + position-try-fallbacks

## position-try-order

для позиционирования

```scss
.position-try-order {
  position-try-order: normal;
  position-try-order: most-height;
  position-try-order: most-width;
  position-try-order: most-block-size;
  position-try-order: most-inline-size;
}
```

## position-try-fallbacks

резервные позиции для размещения

```scss
.position-try-fallbacks {
  position-try-fallbacks: flip-block;
  position-try-fallbacks: top;
  position-try-fallbacks: --custom-try-option;

  /* Multiple value combination option */
  position-try-fallbacks: flip-block flip-inline;

  /* Multiple values */
  position-try-fallbacks: flip-block, flip-inline;
  position-try-fallbacks: top, right, bottom;
  position-try-fallbacks: --custom-try-option1, --custom-try-option2;
  position-try-fallbacks: flip-block, flip-inline, flip-block flip-inline;
  position-try-fallbacks: flip-block, --custom-try-option,
    --custom-try-option flip-inline, right;
}
```

# position-visibility

отвечает за отражение

# @position-try

Позволяет расположить якорь. Нет в сафари и в ff

```html
<div class="anchor">⚓︎</div>

<div class="infobox">
  <p>This is an information box.</p>
</div>
```

```scss
.anchor {
  anchor-name: --myAnchor;
  position: absolute;
  top: 100px;
  left: 350px;
}

@position-try --custom-left {
  inset-area: left;
  width: 100px;
  margin: 0 10px 0 0;
}

@position-try --custom-bottom {
  top: anchor(bottom);
  justify-self: anchor-center;
  margin: 10px 0 0 0;
  inset-area: none;
}

@position-try --custom-right {
  left: calc(anchor(right) + 10px);
  align-self: anchor-center;
  width: 100px;
  inset-area: none;
}

@position-try --custom-bottom-right {
  inset-area: bottom right;
  margin: 10px 0 0 10px;
}

.infobox {
  position: fixed;
  position-anchor: --myAnchor;
  inset-area: top;
  width: 200px;
  margin: 0 0 10px 0;
  position-try-fallbacks: --custom-left, --custom-bottom, --custom-right,
    --custom-bottom-right;
}
```

# anchor()

для определения позиции якоря

```scss
 {
  /* side or percentage */
top: anchor(bottom);
top: anchor(50%);
top: calc(anchor(bottom) + 10px)
inset-block-end: anchor(start);

/* side of named anchor */
top: anchor(--myAnchor bottom);
inset-block-end: anchor(--myAnchor start);

/* side of named anchor with fallback */
top: anchor(--myAnchor bottom, 50%);
inset-block-end: anchor(--myAnchor start, 200px);
left: calc(anchor(--myAnchor right, 0%) + 10px);
}
```

# anchor-size()

функция для измерения якоря

```scss
 {
  width: anchor-size(width);
  block-size: anchor-size(block);
  height: calc(anchor-size(self-inline) + 2em);

  /* size of named anchor side */
  width: anchor-size(--myAnchor width);
  block-size: anchor-size(--myAnchor block);

  /* size of named anchor side with fallback */
  width: anchor-size(--myAnchor width, 50%);
  block-size: anchor-size(--myAnchor block, 200px);
}
```

# text-anchor

выравнивает блок, содержащий строку текста, где область переноса определяется из свойства

Пример со слайдером, в котором будет отображаться значение шкалы

```html
<label for="slider">Change the value:</label>
<input type="range" min="0" max="100" value="25" id="slider" />
<output>25</output>
<script>
  const input = document.querySelector("input");
  const output = document.querySelector("output");

  input.addEventListener("input", (event) => {
    output.innerText = `${input.value}`;
  });
</script>
```

```scss
input::-webkit-slider-thumb {
  // цепляемся за слайдер
  anchor-name: --thumb;
}

output {
  // цепляемся за слайдер
  position-anchor: --thumb;
  position: absolute;
  left: anchor(right);
  bottom: anchor(top);
}
```
