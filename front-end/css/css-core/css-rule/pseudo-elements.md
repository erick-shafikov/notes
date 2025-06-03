# Псевдоклассы

Псевдоэлементы позволяют задать стиль элементов не определенных в дереве элементов документа, а также сгенерировать содержимое, которого нет в исходном коде текста.

Список всех элементов:

## ::after и ::before

для вставки назначенного контента после содержимого элемента, работает совместно со стилевым свойством content которое определяет содержимое вставки. Добавляет последним потомка

```scss
p.new:after {
  content: "-Новьё!";
}
```

```html
<p class="new"></p>
```

[Совмещение с пользовательскими data- атрибутами](#bp-подсказка-с-помощью-after)

## ::backdrop

работает в паре с fullscreenApi и dialog.

## ::content

заменяет элемент сгенерированным значением

```scss
.elem:after {
  content: normal;
  content: none;

  /* Значение <url>  */
  content: url("http://www.example.com/test.png");

  /* Значение <image>  */
  content: linear-gradient(#e66465, #9198e5);

  /* Указанные ниже значения могут быть применены только к сгенерированному контенту с использованием ::before и ::after */

  /* Значение <string>  */
  content: "prefix";

  /* Значения <counter> */
  content: counter(chapter_counter);
  content: counters(section_counter, ".");

  /* Значение attr() связано со значением атрибута HTML */
  content: attr(value string);

  /* Значения кавычек */
  content: open-quote;
  content: close-quote;
  content: no-open-quote;
  content: no-close-quote;

  /* Несколько значений могут использоваться вместе */
  content: open-quote chapter_counter;
}
```

Пример с возможность заменить

```scss
#replaced {
  content: url("mdn.svg");
}

#replaced::after {
  /* не будет отображаться, если замена элемента поддерживается */
  content: " (" attr(id) ")";
}
```

## ::cue

в медиа с VTT треками, создан для создания титров

## ::details-content (-ff, -safari)

представляет контент дял details

```html
<details>
  <summary>Click me</summary>
  <p>Here is some content</p>
</details>
```

```scss
details::details-content {
  background-color: #a29bfe;
}
```

## ::file-selector-button

кнопка выбора фала input type === file

```scss
input[type="file"]::file-selector-button {
  border: 2px solid #6c5ce7;
  padding: 0.2em 0.4em;
  border-radius: 0.2em;
  background-color: #a29bfe;
  transition: 1s;
}

input[type="file"]::file-selector-button:hover {
  background-color: #81ecec;
  border: 2px solid #00cec9;
}
```

## ::first-letter

Определяет стиль первого символа в тексте элемента

## ::first-line

определяет стиль первой строчки блочного текста

## ::grammar-error (-ff)

представляет сегмент текста, который user agent пометил как грамматически неверный.

## ::highlight()

для выделения текста

## ::selection

для выделенной части

## ::marker (-s)

маркер списка

## ::placeholder (-s)

для input текста placeholder,

## ::spelling-error (-ff)

## ::target-text

для прокрученного текста

<!-- Экспериментальные ----------------------------------------------------------------------------------------------------------------------->

# Экспериментальные:

- ::backdrop - это прямоугольник с размерами окна, который отрисовывается сразу же после отрисовки любого элемента в полноэкранном режиме,

<!-- view-transition ------------------------------------------------------------------------------------------------------------------------->

# Используемые для view-transition

## ::view-transition

верхний элемент переходов

```scss
html::view-transition {
  position: fixed;
  inset: 0;
}
```

```scss
::view-transition {
  background-color: rgb(0 0 0 / 25%);
}
```

## ::view-transition-group

отдельная группа

## ::view-transition-image-pair

для "old" и "new" состояний

## ::view-transition-new

новая стадия перехода, состояние после перехода, потомок ::view-transition-image-pair

## ::view-transition-old

изначальная стадия, до перехода

- [свойство content](./css-props.md/#content)

<!-- кастомные элементы ------------------------------------------------------------->

# кастомные элементы

## ::part()

## ::slotted

дял помещенных в слот

<!-- moz ------------------------------------------------------------------------------------------------------------------------------------->

# moz

Работают только в ff

- ::-moz-color-swatch - стилизация компонента input type color
- ::-moz-focus-inner - стилизация кнопок при фокусе
- ::-moz-list-bullet - стилизация маркера списка
- ::-moz-list-number - стилизация цифры нумерованного списка
- ::-moz-meter-bar - стилизация элемента meter
- ::-moz-progress-bar - стилизация элемента progress
- ::-moz-range-progress - стилизация элемента range, саму шкалу
- ::-moz-range-thumb - стилизация элемента range, ползунок
- ::-moz-range-track - стилизация элемента range, саму шкалу

<!-- webkit ---------------------------------------------------------------------------------------------------------------------------->

# webkit (-ff)

- ::-webkit-inner-spin-button - стилизация компонента input type number (переключателей - увеличение и уменьшения значений)
- стилизация компонента meter:
- - ::-webkit-meter-bar - стилизация элемента meter
- - ::-webkit-meter-even-less-good-value - стилизация элемента meter с учетом min и nax аттрибутов
- - ::-webkit-meter-inner-element - стилизация элемента meter, контейнер
- - ::-webkit-meter-optimum-value - meter, Значение находится от min до max
- - ::-webkit-meter-suboptimum-value - меньше оптимального значения
- стилизация компонента progress:
- - ::-webkit-progress-bar - стилизация незаполненной части
- - ::-webkit-progress-inner-element - стилизация контейнера
- - ::-webkit-progress-value - стилизация заполненной части
- ::-webkit-scrollbar
- - ::-webkit-scrollbar - вся полоса прокрутки.
- - ::-webkit-scrollbar-button - кнопки на полосе прокрутки (стрелки, направленные вверх и вниз, которые прокручивают по одной строке за раз).
- - ::-webkit-scrollbar:horizontal - горизонтальная полоса прокрутки.
- - ::-webkit-scrollbar-thumb - перетаскиваемый маркер прокрутки.
- - ::-webkit-scrollbar-track - полоса прокрутки (прогресс-бар), где поверх белой полосы находится серая полоса.
- - ::-webkit-scrollbar-track-piece - часть дорожки (прогресс-бара), не охваченная ручкой.
- - ::-webkit-scrollbar:vertical - вертикальная полоса прокрутки.
- - ::-webkit-scrollbar-corner - нижний угол полосы прокрутки, где встречаются горизонтальная и вертикальная полосы прокрутки. Часто это нижний правый угол окна браузера.
- - ::-webkit-resizer - перетаскиваемый маркер изменения размера, который появляется в нижнем углу некоторых элементов.
- ::-webkit-search-cancel-button - type="search"
- ::-webkit-search-results-button - если есть results атрибут
- input type range
- - ::-webkit-slider-runnable-track - input type range
- - ::-webkit-slider-thumb

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs

## BP. Иконка меню

```css
.nav-btn {
  border: none;
  border-radius: 0;
  background-color: #fff;
  height: 2px;
  width: 4.5rem;
  margin-top: 4rem;
  /*элементы до и после   */
  &::before,
  &::after {
    content: "";
    display: block;
    background-color: #fff;
    height: 2px;
    width: 4.5rem;
  }
  /* располагаем     */
  &::before {
    transform: translateY(-1.5rem);
  }
  &::after {
    transform: translateY(1.3rem);
  }
}
```

## BP. подсказка с помощью after

```scss
// для всех span у которых есть атрибут descr
span[data-descr] {
  //позиционируем relative
  position: relative;
  // стилизуем текст
  text-decoration: underline;
  color: #00f;
  cursor: help;
}

// при hover
span[data-descr]:hover::after {
  // берем из атрибута текст
  content: attr(data-descr);
  // позиционируем
  position: absolute;
  left: 0;
  top: 24px;
  min-width: 200px;
  border: 1px #aaaaaa solid;
  border-radius: 10px;
  background-color: #ffffcc;
  padding: 12px;
  color: #000000;
  font-size: 14px;
  z-index: 1;
}
```

```html
<p>
  Здесь находится живой пример вышеприведённого кода.<br />
  У нас есть некоторый
  <span data-descr="коллекция слов и знаков препинаний">текст</span> здесь с
  несколькими
  <span data-descr="маленькие всплывающие окошки, которые снова исчезают"
    >подсказками</span
  >.<br />
  Не стесняйтесь, наводите мышку чтобы
  <span data-descr="не понимать буквально">взглянуть</span>.
</p>
```
