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
