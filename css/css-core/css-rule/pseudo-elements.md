Псевдоэлементы позволяют задать стиль элементов не определенных в дереве элементов документа, а также сгенерировать содержимое, которого нет в исходном коде текста.

Список всех элементов:

- ::after – для вставки назначенного контента после содержимого элемента, работает совместно со стилевым свойством content которое определяет содержимое вставки, часто используют со свойством content. Добавляет последним потомка

```scss
p.new:after {
  content: "-Новьё!";
}
```

```html
<p class="new"></p>
```

- ::cue - в медиа с VTT треками
- ::file-selector-button - кнопка выбора фала input type === file
- ::first-letter Определяет стиль первого символа в тексте элемента
- ::first-line определяет стиль первой строчки блочного текста
- ::selection – для выделенной части
- ::slotted - дял помещенных в слот,
- ::marker - маркер списка (нет в safari)

Экспериментальные:

- ::backdrop - это прямоугольник с размерами окна, который отрисовывается сразу же после отрисовки любого элемента в полноэкранном режиме,
- ::placeholder (нет в Safari) - для input текста placeholder,
- ::marker (нет в Firefox) - поле маркера списка,
- ::spelling-error (нет в Firefox),
- ::grammar-error (нет в Firefox) - элемент, который имеет грамматическую ошибку

Используемые для view-transition

- ::view-transition - верхний элемент переходов
- ::view-transition-group - отдельная группа
- ::view-transition-image-pair - "old" and "new"
- ::view-transition-new - новая стадия перехода
- ::view-transition-old - изначальная стадия

- [свойство content](./css-props.md/#content)

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
