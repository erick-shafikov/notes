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

# print-color-adjust (нет в chrome, edge)

для устройств вывода

```scss
.print-color-adjust {
  print-color-adjust: economy; //Пользовательскому агенту разрешено вносить изменения в элемент, которые он считает целесообразными и разумными
  print-color-adjust: exact;
}
```

# @media(prefers-color-scheme)

- [медиа запросы для различных тем](./at-rules.md/#mediaprefers-color-scheme)

## mix-blend-mode

определяет режим смешивания цветов выбранного элемента с низлежащими слоями.

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

# isolation

Управление контекстом контекст наложения,

```scss
 {
  isolation: auto; //цвета ниже будут пробиваться
  isolation: isolate; //цвета ниже не будут пробиваться
}
```
