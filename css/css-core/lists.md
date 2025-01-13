Списки имеют предустановленные стили

При создания списка у элюентов li display: list-item

# line-height

[для расстояния между li](./text.md#line-height)

# list-style:

для стилизации маркеров списка

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

# @counter-style

изменение маркеров @counter-style позволяет определить отображение

```scss
@counter-style symbols-example {
  system: cyclic | numeric | ....; //https://developer.mozilla.org/en-US/docs/Web/CSS/@counter-style/system
  symbols: A "1" "\24B7"D E; //ряд символов на повторение
  additive-symbols: 1000 M, 900 CM, 500 D, 400 CD, 100 C, 90 XC, 50 L, 40 XL,
  symbols: url(gold-medal.svg) url(silver-medal.svg) url(bronze-medal.svg); //могут быть и изображения
    10 X, 9 IX, 5 V, 4 IV, 1 I; //позволяет задать ряд с системой исчисления
  negative: "--"; // задать элементы, если они начинаются с отрицательного индекса атрибут start < 0
  prefix: "»";
  suffix: "";
  range: 2 4, 7 9; //на какие по счету элементы будет применяться
  pad: 3 "0";
  speak-as: auto |...;
  fallback: lower-alpha; //альтернативный
}

.items {
  list-style: symbols-example;
}
```

Пример 2

```scss
@counter-style options {
  system: fixed;
  symbols: A B C D;
  suffix: ") ";
}

.choices {
  list-style: options;
}
```

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

# counter() counter-reset() counter-increment()

активирует запуск счетчика на элементах

```scss
.double-list {
  counter-reset: count -1;
}

.double-list li {
  counter-increment: count 2;
}

.double-list li::marker {
  content: counter(count, decimal) ") ";
}
```

создаст список с 1-3-5-7-9

```html
<p>Best Dynamic Duos in Sports:</p>
<ol class="double-list">
  <li>Simone Biles + Jonathan Owens</li>
  <li>Serena Williams + Venus Williams</li>
  <li>Aaron Judge + Giancarlo Stanton</li>
  <li>LeBron James + Dwyane Wade</li>
  <li>Xavi Hernandez + Andres Iniesta</li>
</ol>
```

Пример с глобальным счетчиком

```scss
body {
  // Устанавливает значение счётчика, равным 0 переменная - section
  counter-reset: section;
}

h3::before {
  // Инкриминирует счётчик section = ++section
  counter-increment: section;
  // Отображает текущее значение счётчика counter - достает значение
  content: "Секция " counter(section) ": ";
}
```

```html
<h3>Вступление</h3>
<h3>Основная часть</h3>
<h3>Заключение</h3>
```

Вывод
Секция 1: Вступление
Секция 2: Основная часть
Секция 3: Заключение

# counters()

для вложенных списков

```scss
ol {
  counter-reset: index;
  list-style-type: none;
}

li::before {
  counter-increment: index;
  content: counters(index, ".", decimal) " ";
}
```

```scss
ol {
  counter-reset: section; /* Создаёт новый счётчик для каждого тега <ol> */
  list-style-type: none;
}

li::before {
  counter-increment: section; /* Инкриминируется только счётчик текущего уровня вложенности */
  // уже не counter
  content: counters(section, ".") " "; /* Добавляем значения всех уровней вложенности, используя разделитель '.' */
  /* Если необходима поддержка < IE8, необходимо убедиться, что после разделителя ('.') не стоит пробел */
}
```
