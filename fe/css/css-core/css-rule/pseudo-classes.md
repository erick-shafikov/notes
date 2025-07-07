псевдокласс - это селектор, который выбирает элементы находящиеся в специфическом состоянии. Псевдоклассы определяют динамическое состояние элементов, они находят что-то внутри тега, конкретизирует синтаксис

<!-- выбор элементов DOM --------------------------------------------------------------------------------------------------------------------->

# выбор элементов DOM

## :empty

находит любой элемент, у которого нет потомков,

## :first-child

находит любой элемент, являющийся первым в своём родителе

```scss
// найди все p которые являются первыми вложенными
p:first-child {
  background-color: red;
}
```

```html
<div>
  <p>Применится к этому элементу, так как это первый p в своем родителе</p>
  <p>Не применится так как это второй</p>
</div>

<div>
  <h2></h2>
  <p>Не применится так как это второй</p>
</div>
```

## :first-of-type

находит первого потомка своего типа среди детей родителя, то есть первые типы тегов из всех дочерних, то есть первого ребенка другого типа

```scss
// найди все p которые являются первыми вложенными в кого-либо
div :first-of-type {
  background-color: lime;
}
```

```html
<div>
  <span>Применится так как первый ребенок div</span>
  <span>не применится так как второй</span>
  <span>не применится <em>Применится так как первый ребенок из em</em>?</span>
  <strike>Применится так как первый ребенок из strike</strike>
  <span>не применится</span>
</div>
```

## :has()

если хотя бы один из относительных селекторов, переданных в качестве аргумента, соответствует хотя бы одному элементу.

```scss
/* Отступ снизу будет обнулён только для тегов <h1>,
следом за которыми идёт тег <p> */
h1:has(+ p) {
  margin-bottom: 0;
}
```

```scss
//Следующий селектор находит только те теги <a>, которые непосредственно содержат дочерний элемент <img>
a:has(> img) {
}
```

## :is()

любой селектор из списка совместим с :matches(), :any()

```scss
// позволяет преобразовать из такого набора селекторов
header p:hover,
main p:hover,
footer p:hover {
  color: red;
  cursor: pointer;
}

// в такой
// для поддержки
// :matches(header, main, footer) p:hover
// :-moz-any(header, main, footer) p:hover
// :-webkit-any(header, main, footer) p:hover
:is(header, main, footer) p:hover {
  color: red;
  cursor: pointer;
}
```

Проверки на соответствия позволяют упростить селекторы на 3х и более уровнях вложенности в таком случае как ol ul li в разном порядке:

```scss
// из такого с многоуровневой вложенностью
ol ol ul,
ol ul ul,
ol menu ul,
ol dir ul,
ol ol menu,
ol ul menu,
ol menu menu,
ol dir menu,
ol ol dir,
ol ul dir,
ol menu dir,
ol dir dir,
ul ol ul,
ul ul ul,
ul menu ul,
ul dir ul,
ul ol menu,
ul ul menu,
ul menu menu,
ul dir menu,
ul ol dir,
ul ul dir,
ul menu dir,
ul dir dir,
menu ol ul,
menu ul ul,
menu menu ul,
menu dir ul,
menu ol menu,
menu ul menu,
menu menu menu,
menu dir menu,
menu ol dir,
menu ul dir,
menu menu dir,
menu dir dir,
dir ol ul,
dir ul ul,
dir menu ul,
dir dir ul,
dir ol menu,
dir ul menu,
dir menu menu,
dir dir menu,
dir ol dir,
dir ul dir,
dir menu dir,
dir dir dir {
  list-style-type: square;
}

// в такой
:is(ol, ul, menu, dir) :is(ol, ul, menu, dir) ul,
:is(ol, ul, menu, dir) :is(ol, ul, menu, dir) menu,
:is(ol, ul, menu, dir) :is(ol, ul, menu, dir) dir {
  list-style-type: square;
}
```

- если один из селекторов не поддерживает - все правило НЕ сбросится
- Отличает от where - у where специфичность === 0

## last-child

последний из элементов данного типа в родителе

```scss
li:last-child {
  background-color: lime;
}
```

```html
<ul>
  <li>не применится</li>
  <li>не применится<</li>
  <li>применится<</li>
</ul>
```

## :last-of-type

выберет последний тег,

```scss
p:last-of-type {
  color: red;
  font-style: italic;
}
```

```html
<h2>Нет</h2>
<p>Нет</p>
<p>Да</p>
```

## :not()

принимает селектор

```scss
// p без класса classy
p:not(.classy) {
  color: red;
}

// не p внутри body
body :not(p) {
  color: green;
}
```

## :nth-child()

находит один или более элементов, основываясь на их позиции среди группы соседних элементов,

Значения аргумента - odd, even, формула An + B, где n - позиция начиная с 1

- tr:nth-child(odd) или tr:nth-child(2n+1) - Описывает нечётные строки HTML таблицы: 1, 3, 5, и т. д.

- tr:nth-child(even) or tr:nth-child(2n) - Описывает чётные строки HTML таблицы: 2, 4, 6, и т. д.

- :nth-child(7) - Описывает седьмой элемент.

- :nth-child(5n) - Описывает элементы с номерами 5, 10, 15, и т. д.

- :nth-child(3n+4) - Описывает элементы с номерами 4, 7, 10, 13, и т. д.

- :nth-child(-n+3) - Описывает первые три элемента среди группы соседних элементов.

- p:nth-child(n) - Описывает каждый элемент <p> среди группы соседних элементов. Эквивалентно простому селектору p.

- p:nth-child(1) или p:nth-child(0n+1) Описывает каждый элемент p, являющийся первым среди группы соседних элементов. Эквивалентно селектору :first-child.

## :nth-last-child()

an+b-1 элемент,

## :nth-last-of-type()

последний элемент с заданным тегом

## :nth-of-type()

находит один или более элементов с заданным тегом,

## :only-child

дял элемента, который является единственным предком

## :only-of-type

выбирает такой элемент, который является единственным потомком такого типа,

- :target - если он (его id) является целью текущего url,

```scss
.lightbox {
  // изначально скрыт
  display: none;
}

/* Открываем lightbox */
.lightbox:target {
  // как станет целевым
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
}

/* Содержимое lightbox  */
.lightbox figcaption {
}

/* Кнопка закрытия */
.lightbox .close {
}

// иконка закрытия
.lightbox .close::after {
  content: "×";
  cursor: pointer;
}

/* Обёртка lightbox  */
.lightbox .close::before {
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  position: fixed;
  background-color: rgba(0, 0, 0, 0.7);
  content: "";
  cursor: default;
}
```

```html
<ul>
  <!-- откроет одно модально окно -->
  <li><a href="#example1">Open example #1</a></li>
  <!-- откроет второе -->
  <li><a href="#example2">Open example #2</a></li>
</ul>

<!-- скрытые окна -->
<div class="lightbox" id="example1">
  <figure>
    <!-- кнопка закрытия, как будет нажата example1 или example2 потеряют псевдокласс target-->
    <a href="#" class="close"></a>
    <figcaption></figcaption>
  </figure>
</div>

<div class="lightbox" id="example2">
  <figure>
    <a href="#" class="close"></a>
    <figcaption></figcaption>
  </figure>
</div>
```

## :target-within

если элемент является target или включает в себя target

```scss
// выбрать div если один из потомков является target
div:target-within {
  background: cyan;
}
```

## :where()

:is(), :any() только с нулевой специфичностью

## BP. Выбор диапазона элементов

```scss
.elem:nth-child(n + 3):nth-last-child(n + 3) {
  //выбрать первые три и последние три
}
```

<!-- прочие ---------------------------------------------------------------------------------------------------------------------------------->

# прочие

## :dir()

выбирает элементы на основе направления текста :dir(rtl){...},

## :fullscreen

соответствует элементу, который в данный момент находится в полноэкранном режиме, (нет в safari)

## :lang(en|)

элемент:lang(язык) на элементах должен быть определен атрибут lang

полезно для определения кавычек, для контента обернутого в q

```scss
:lang(en) > q {
  quotes: "\201C""\201D""\2018""\2019";
}
:lang(fr) > q {
  quotes: "« " " »";
}
:lang(de) > q {
  quotes: "»" "«" "\2039""\203A";
}
:lang(en) > q {
  quotes: "\201C""\201D""\2018""\2019";
}
:lang(fr) > q {
  quotes: "« " " »";
}
:lang(de) > q {
  quotes: "»" "«" "\2039""\203A";
}
```

## :picture-in-picture

для элементов в picture-in-picture mode

## :root

элемент, который является корнем документа используется для определения переменных то есть тег html,

- :scope - для элементов области видимости, может выступать альтернативе root

```html
<div class="light-scheme">
  <p>
    MDN contains lots of information about
    <a href="/en-US/docs/Web/HTML">HTML</a>,
    <a href="/en-US/docs/Web/CSS">CSS</a>, and
    <a href="/en-US/docs/Web/JavaScript">JavaScript</a>.
  </p>
</div>

<div class="dark-scheme">
  <p>
    MDN contains lots of information about
    <a href="/en-US/docs/Web/HTML">HTML</a>,
    <a href="/en-US/docs/Web/CSS">CSS</a>, and
    <a href="/en-US/docs/Web/JavaScript">JavaScript</a>.
  </p>
</div>
```

```scss
@scope (.light-scheme) {
  :scope {
    background-color: plum;
  }

  a {
    color: darkmagenta;
  }
}

@scope (.dark-scheme) {
  :scope {
    background-color: darkmagenta;
    color: antiquewhite;
  }

  a {
    color: plum;
  }
}
```

<!-- BP -------------------------------------------------------------------------------------------------------------------------------------->

# BPs

## BP. Стилизация ссылки с помощью псевдоклассов

```scss
// стилизация всех возможных состояний ссылки
&__link {
  &:link,
  &:visited {
  }

  &:hover,
  &:active {
  }
}
```

## BP. Модальное окно с помощью псевдоклассов

1. Ссылка будет направлять на id в href

```html
<a href="#popup">Book now!</a>
```

2. Прописать стили

```scss
&:target {
  opacity: 1;
  visibility: visible;
}
//при клике станет прозрачным
&:target &__content {
  //становится не прозрачным
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
}
```

3. реализовать кнопку закрытия

```html
<a href="#section-tours" class="popup__close">&times;</a>
```
