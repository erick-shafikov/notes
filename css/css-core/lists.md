<!-- Списки --------------------------------------------------------------------------------------------------------------------------------->

# Списки

Списки имеют предустановленные стили

При создания списка у элюентов li display: list-item

- [line-height - для расстояния между li](./text.md#line-height)
- list-style - для стилизации маркеров списка

Сокращенная запись для list-style = list-style-image + list-style-position + list-style-type

```scss
 {
  //тип маркеров
  list-style-type: disc;
  list-style-type: circle;
  list-style-type: square;
  list-style-type: decimal;
  list-style-type: georgian;
  list-style-type: trad-chinese-informal;
  list-style-type: kannada;
  list-style-type: "-";
  /* Identifier matching an @counter-style rule */
  list-style-type: custom-counter-style;
  list-style-type: none;
  list-style-type: inherit;
  list-style-type: initial;
  list-style-type: revert;
  list-style-type: revert-layer;
  list-style-type: unset;

  //где будет располагаться
  list-style-position: inside; //::marker перед контентом
  list-style-position: outside; //::marker внутри контента

  //изображение
  list-style-image: url(example.png);

  // шорткат
  list-style: square url(example.png) inside; // list-style-type list-style-image list-style-position
}
```

## list-style-image

Позволяет добавить изображение в список в качестве разделителя

```scss
 {
  list-style-image: none;

  /* <url> значения */
  list-style-image: url("starsolid.gif");
}
```

## list-style-position

inside | outside - расположение маркера внутри отступа или вне отступа

# счетчики

[изменение маркеров @counter-style](./at-rules.md)

# counter-increment, counter-set, counter-reset,

используется для увеличения значений в списке

```scss
// сброс счетчика
div {
  counter-reset: my-counter 100; //задает новое значение
}
div {
  // объявляем счетчик и начальное значение по умолчанию ноль
  counter-increment: my-counter -1;
}
div {
  // объявляем счетчик и начальное значение по умолчанию ноль
  counter-set: my-counter -1; //задает новое значение
}
i::before {
  // запуск c помощью функции counter
  content: counter(sevens);
}
```

-[функция counter()](./functions.md)

список, который уменьшается на 1

```html
<div>
  <i>1</i>
  <i>100</i>
</div>
```

- - [функции которые позволяют формировать нумерованные списки](./functions.md/#counter-counter-reset-counter-increment)
- - - [вложенные нумерованные списки](./functions.md/#counters)
