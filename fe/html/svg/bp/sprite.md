# Вставка svg-спрайта

```html
<!-- Вставка svg-спрайта-->
<svg class="feature__icon">
  <use xlink:href="img/sprite.svg#icon-global"></use>
</svg>
```

каждая иконка идет как symbol с уникальным id

```html
<svg xmlns="<http://www.w3.org/2000/svg>">
  <symbol viewBox="0 0 24 24" id="check">
    <path
      d="M20 6L9 17L4 12"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </symbol>
  <symbol viewBox="0 0 24 24" id="close">
    <path
      d="M17 7L7 17M7 7L17 17"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </symbol>
</svg>
```

использование

```html
<div>
  <svg width="20" height="20">
    <use href="/sprite.svg#check"></use>
  </svg>
</div>
```
