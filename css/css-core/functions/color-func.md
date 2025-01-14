<!-- работа с цветами ---------------------------------------------------------------------------------------------------------------------------->

# работа с цветами

- color-contrast()
- color-mix() - для смешивания двух цветов
- color() - Позволяет задавать цветовые пространства

```html
<div data-color="red"></div>
<div data-color="green"></div>
<div data-color="blue"></div>
```

```scss
[data-color="red"] {
  background-color: color(xyz 45 20 0);
}

[data-color="green"] {
  background-color: color(xyz-d50 0.3 80 0.3);
}

[data-color="blue"] {
  background-color: color(xyz-d65 5 0 50);
}
```

- device-cmyk()
- hsl()
- hwb()
- lab()
- lch()
- oklab()
- oklch()

- rgb()

<!-- Градиенты  ---------------------------------------------------------------------------------------------------------------------------->

# градиенты
