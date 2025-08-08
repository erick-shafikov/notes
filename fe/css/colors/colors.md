# color-scheme

Применит стили темы пользователя для элементов

```scss
.color-scheme {
  color-scheme: normal;
  color-scheme: light; //Означает, что элемент может быть отображён в светлой цветовой схеме операционной системы.
  color-scheme: dark; //Означает, что элемент может быть отображён в тёмной цветовой схеме операционной системы.
  color-scheme: light dark;
}

:root {
  color-scheme: light dark;
}
```

# forced-color-adjust

auto | none позволяет включить и выключить изменения тема в режиме высокого контраста

# print-color-adjust (-chrome, -edge)

для устройств вывода

```scss
.print-color-adjust {
  print-color-adjust: economy; //Пользовательскому агенту разрешено вносить изменения в элемент
  print-color-adjust: exact;
}
```

```scss
.print-color-adjust {
  print-color-adjust: economy; //Пользовательскому агенту разрешено вносить изменения в элемент, которые он считает целесообразными и разумными
  print-color-adjust: exact;
}
```

# isolation

Управление контекстом контекст наложения,

```scss
 {
  isolation: auto; //цвета ниже будут пробиваться
  isolation: isolate; //цвета ниже не будут пробиваться
}
```

# opacity

Прозрачность элемента

```scss
div {
  background-color: yellow;
}
.light {
  opacity: 0.2; /* Едва видимый текст на фоне */
}
.medium {
  opacity: 0.5; /* Видимость текста более чёткая на фоне */
}
.heavy {
  opacity: 0.9; /* Видимость текста очень чёткая на фоне */
}
```

# @media(prefers-color-scheme)

- [медиа запросы для различных тем](./at-rules.md/#mediaprefers-color-scheme)

## mix-blend-mode

определяет режим смешивания цветов выбранного элемента с нижележащими слоями.

```scss
.mix-blend-mode {
  mix-blend-mode: normal;
  mix-blend-mode: multiply;
  mix-blend-mode: screen;
  mix-blend-mode: overlay;
  mix-blend-mode: darken;
  mix-blend-mode: lighten;
  mix-blend-mode: color-dodge;
  mix-blend-mode: color-burn;
  mix-blend-mode: hard-light;
  mix-blend-mode: soft-light;
  mix-blend-mode: difference;
  mix-blend-mode: exclusion;
  mix-blend-mode: hue;
  mix-blend-mode: saturation;
  mix-blend-mode: color;
  mix-blend-mode: luminosity;
}
```

<!-- @color-profile -------------------------------------------------------------------------------------------------------------------------->

# @color-profile

Определяет цветовой профиль

```scss
// имя --swop5c
@color-profile --swop5c {
  src: url("https://example.org/SWOP2006_Coated5v2.icc");
}

.header {
  background-color: color(--swop5c 0% 70% 20% 0%);
}
```

<!-- функции по работе с цветами ------------------------------------------------------------------------------------------------------------->

# функции по работе с цветами

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
