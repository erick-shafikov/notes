```scss
//селектор: hover//псевдокласс
a {
  //свойство
  background-color: #f8f8f8 //значение
;
}
```

Типы значений: integer, number, dimension, percentage, color

Единицы измерения:

- ch - глифа 0 шрифта элемента
- em - Размер шрифта родительского элемента.
- ex - x высота шрифта
- lh - высота строки элемента
- rem (root em) - относительно корня, html, у которого задан font-size
- px
- vh - 1% от высоты
- vw - 1% от ширины
- vmin/vmax - 1% от меньшего/большего ширины окна
- процент
- - !!! margin и padding могут быть в процентах. Рассчитывается на основе Inline Блока
- числа (от 0 до 1)

Цвета: 16х, RGB, RGBA, hsl, hsla

```scss
// пример с rgba
.one {
  background-color: rgb(2 121 139, 0.3);
}
```

селектор – это имя стиля, для которого добавляют элементы форматирования. В качестве селекторы выступают классы и идентификаторы

При необходимости установить одновременно один стиль для всех элементов веб-страницы, например задать шрифт или начертание текста. Синтаксис:

- {описание стиля}

# Селекторы

Существует более 80 селекторов и комбинаторов.

Базовый селекторы:

- [универсальный](#универсальный-селектор)
- [по типу элемента](#селектор-по-типу-элемента)
- [селектор по классу](#селектор-по-классу)
- [селектор по идентификатору](#селектор-по-идентификатору)
- [селектор по аттрибуту](#селектор-по-атрибуту)

!!! Если в правиле один из селекторов неправильно, то правило не будет применимо

## Универсальный селектор

```
ns|* - вхождения всех элементов в пространстве имён ns
*|* - находит все элементы
|* - ищет все элементы без объявленного пространства имён
```

```scss
*[lang^="en"] {
  color: green;
}
*.warning {
  //*.warning === .warning
  color: red;
}
*#maincontent {
  border: 1px solid blue;
}
```

## Селектор по типу элемента

```scss
span {
  background-color: DodgerBlue;
}
```

## Селектор по классу

Названия классов могут содержать нижнее подчеркивание и дефис

```scss
.class-name {
  // Селектор по классу
}
[class~="class-name"] {
  //  эквивалентная запись
}
```

```html
<p><span class="ghost">текст</span><span class="term">контент</span></p>
```

## Селектор по идентификатору

```scss
[id="id_value"] {
  // эквивалентная запись
}

span#identified {
  background-color: DodgerBlue;
}
```

## Селектор по атрибуту

Если добавить стиль атрибуту, то он применится ко всем тегам. Что бы сделать гибким управление используют атрибуты

- [attr] ищет элементы с указанным атрибутом
- [attr=value] Ищет элементы с attr === value полное совпадение

```html
<style>
  A[target="_blank"] {
    /* параметры фонового рисунка */
    background: url(images/blank.png) 0 6px no-repeat;
    /* смещает текст вправо */
    padding-left: 15px;
  }
</style>
<!-- линия появится в документе -->
<p>
  <a href="1.html"> обычная ссылка </a> |
  <a href="link" target="_blank">Ссылка в новом окне</a>
</p>
```

- [атрибут^="значение"] Значение атрибута начинается с определенного текста

Нужно разделить стиль обычных и внешних ссылок

```html
<style>
  A[href^="http://"]
  {
    front-weight: bold; /*Жирное начертание*/
  }
</style>
<p>
  <a href="1.html"> Обычная ссылка </a>
  <a href="http://..." target="blank">Внешняя ссылка</a>
</p>
```

- Значение атрибута заканчивается определенным текстом [атрибут$="значение"]
- Значение атрибута содержит указанный текст [атрибут*="значение"]
- [атрибут~="значение"] Обозначает элемент с именем атрибута attr значением которого является набор слов разделённых пробелами, одно из которых в точности равно value
- [атрибут|="значение"] Обозначает элемент с именем атрибута attr. Его значение при этом может быть или в точности равно "value" или может начинаться с "value" со сразу же следующим "-"

Если в селекторе указать i то поиск будет без учета регистра [атрибут~="значение" i]

# Комбинаторы

Комбинаторы

- [запятая – группировка](#комбинатор-запятая-)
- [пробел – выбор всех потомков](#селектор-потомка-пробел)
- [> - дочерние (непосредственно прямые потомки)](#дочерний-комбинатор--знак-больше)
- [~ - выбор всех одноуровневых](#комбинатор-всех-соседних--тильда)
- [+ - выбор первого соседнего элемента](#соседние-селекторы--знак-плюс)

## Комбинатор запятая ,

Позволяет сгруппировать определение стилей

```html
<style>
  h1 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 160%;
    color: #003;
  }
  h2 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 135%;
    color: #333;
  }
  h3 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 120%;
    color: #900;
  }
  P {
    /* дублирование кода */
    font-family: Times, serif;
  }
</style>
```

Избежать дублирование кода можно

```html
<style>
  h1,
  h2,
  h3 {
    font-family: Arial, Helvetica, sans-serif;
  }
  h1 {
    font-size: 160%;
    color: #003;
  }
  h2 {
    font-size: 135%;
    color: #333;
  }
  h3 {
    font-size: 120%;
    color: #900;
  }
</style>
```

## Селектор потомка (пробел)

Будет применятся для всех вложенных элементов, для всех дочерних

При использовании идентификаторов и классов – позволяет установить стиль внутри определенного класса

```scss
span {
  background-color: white;
}
div span {
  background-color: DodgerBlue;
}
```

```html
<p></p>
<div>
  <span
    >Span 1 (применится)
    <span>Span 2 (применится)</span>
  </span>
</div>
<span>Span 3 (нет)</span>
```

сочетания

```scss
.level11 {
  front-size: 1em;
}
.level12 {
  front-size: 1.2em;
}

a.tag {
  color: #62348;
}
```

```html
<a href="term/2" class="tag level11"> </a>
```

Для комбинации тегов

```html
<style>
  .btn {
    /* для всех кнопок */
  }
  .delete {
    /* дополнительные стили для кнопки удалить */
  }
  .add {
    /*  */
  }
  .edit {
    /*  */
  }
</style>
<button class="btn delete">Удалить</button>
<button class="btn add">Добавить</button>
<button class="btn edit">редактировать</button>
```

## Без пробельный селектор

на смешивание двух классов

```html
<html>
  <head>
  <meta charset=utf-8>
  <title>Камни</title>
  <style>
    /* смешанный селектор */
    table.jewel {
      width: 100%
      border: 1px solid #666;
    }
    th {
      background: #009384;
      color: #fff
      text-align: left;
    }
    tr.odd {
      background: #ebd3d7;
    }
    </style>
  </head>
  <body>
    <!-- применится table.jewel -->
  <table class=jewel>
    <tr>
      <!-- th -->
      <th>Название</th>
    </tr>
    <!-- tr.odd -->
    <tr class=odd>
       <td>Алмаз</td>
    </tr>
</html>
```

!!!Разница с псевдоклассами

```scss
article :first-child {
  // элементы-потомки элемента article
}

article:first-child {
  //выберет любой элемент <article>, являющийся первым дочерним элементом другого элемента
}
```

для классов

```scss
.notebox.danger {
  //
}
```

```html
<div class="notebox danger">This note shows danger!</div>
```

## Дочерний Комбинатор > (знак больше)

выбирает только на первом уровне вложенности

```scss
span {
  background-color: white;
}
div > span {
  background-color: DodgerBlue;
}
```

```html
<div>
  <span
    >Span 1 в div (окрасится)
    <span>Span 2 в span, который в div (не окрасится, не прямой потомок)</span>
  </span>
</div>
<span>Span 3. Не в div вообще</span>
```

## Комбинатор всех соседних ~ тильда

Общий комбинатор смежных селекторов (~) разделяет два селектора и находит второй элемент только если ему предшествует первый, и они оба имеют общего родителя

```scss
p ~ span {
  color: red;
}
```

```html
<span>Это не красный.</span>
<p>Здесь параграф.</p>
<code>Тут какой-то код.</code>
<span
  >А здесь span(применится, так как это первый span который идет сразу после
  p)</span
>
```

## Соседние селекторы + (знак плюс)

выберет непосредственно следующего соседа, с которым он имеет одного родителя

```html
<style>
  b + i {
    /* // все что внутри контейнера I следующего после B будет окрашено в красный цвет */
    color: red;
  }
</style>

<head>
  <meta charset="utf-8" />
  <title>Изменение стиля абзаца</title>
  <style>
    /* //для выделения замечаний */
    H2.sic {
    /* …. */
    }
    H2.sic + P {
    /* … */
    }
  </style>
  <!-- … -->
  <body>
    <h1>Заголовок без стиля</h1>
    <h2>Обычный H2</h2>
    <p>
      …текст без стиля…<p>
        <h2 class="sic">
          …Подобзац со стилем
          <h2>
            <!-- так как тег p идет после H2.sic  -->
            <p>…текст со стилем H2.sic + P</p>
          </h2>
        </h2></p>
      >
    </p>
  </body>
</head>
```

## namespace - селектор namespace|selector

```html
<p>This paragraph <a href="#">has a link</a>.</p>

<svg width="400" viewBox="0 0 400 20">
  <a href="#">
    <text x="0" y="15">Link created in SVG</text>
  </a>
</svg>
```

```scss
@namespace svgNamespace url("http://www.w3.org/2000/svg");
@namespace htmlNameSpace url("http://www.w3.org/1999/xhtml");
/* All `<a>`s in the default namespace, in this case, all `<a>`s */
a {
  font-size: 1.4rem;
}
/* no namespace */
|a {
  text-decoration: wavy overline lime;
  font-weight: bold;
}
/* all namespaces (including no namespace) */
*|a {
  color: red;
  fill: red;
  font-style: italic;
}
/* only the svgNamespace namespace, which is <svg> content */
svgNamespace|a {
  color: green;
  fill: green;
}
/* The htmlNameSpace namespace, which is the HTML document */
htmlNameSpace|a {
  text-decoration-line: line-through;
}
```

# идентификаторы

Идентификатор или ID-селектор определяет уникальное имя элемента

```html
<style>
  #help {
  }
</style>
<div id="help">…контент…</div>

Идентификаторы можно применять к тегу

<style>
  p {
  }
  P#opa {
  }
</style>
<p>…контент…</p>
<p id="opa">…контент…</p>
```

# Псевдоклассы

псевдокласс - это селектор, который выбирает элементы находящиеся в специфическом состоянии. Псевдоклассы определяют динамическое состояние элементов, они находят что-то внутри тега, конкретизирует синтаксис

Проверки на соответствия позволяют упростить селекторы на 3х и более уровнях вложенности в таком случае как ol ul li в разном порядке:

Перечень псевдоклассов:

- :active - при клике на элемент может быть как ссылка так и форма, поля формы,
- :any-link - для всех состояний ссылки,
- :autofill - регулирует поля автозаполнение в полях input
- :blank - для пустого поля ввода или для элемента без потомков
- :checked - для input type="radio" или option внутри select,

```scss
// Находит, все отмеченные на странице, радиокнопки
input[type="radio"]:checked {
}
// входит все отмеченные чекбоксы
input[type="checkbox"]:checked {
}
// Находит все отмеченные option
option:checked {
}
```

Данный псевдокласс позволяет хранить булевские значения в разметке

```scss
#expand-btn {
  //стили чб
}

#isexpanded:checked ~ #expand-btn,
#isexpanded:checked ~ * #expand-btn {
}

#isexpanded,
.expandable {
  // по молчанию скрыты
  display: none;
}

#isexpanded:checked ~ * tr.expandable {
}

#isexpanded:checked ~ p.expandable,
#isexpanded:checked ~ * p.expandable {
  // если появляется checked
  display: block;
}
```

```html
<body>
  <input type="checkbox" id="isexpanded" />
  <table>
    <tbody>
      <tr class="expandable">
        <td>[текст ячейки]</td>
        <td>[текст ячейки]</td>
        <td>[текст ячейки]</td>
      </tr>
      <tr>
        <td>[текст ячейки]</td>
        <td>[текст ячейки]</td>
        <td>[текст ячейки]</td>
      </tr>
      </tr>
    </tbody>
  </table>

  <!-- при клике на label происходит клик по скрытому чб -->
  <label for="isexpanded" id="expand-btn">Показать скрытые элементы</label>
</body>
```

- :current - для стилизации субтитров в видео
- :default - находит элемент формы, установленный по умолчанию, с атрибутами checked
- :defined - работает с пользовательскими элементами объявленные CustomElementRegistry.define(),
- :dir() - выбирает элементы на основе направления текста :dir(rtl){...},
- :disabled -находит любой отключённый элемент с атрибутом disabled
- :empty - находит любой элемент, у которого нет потомков,
- :enabled - находит любой включённый элемент,
- :first - представляя первую страницу документа при печати, [используется с @-правилом @page](./at-rules.md/#page)

```scss
@page :first {
  margin-left: 50%;
  margin-top: 50%;
}
```

- :first-child - находит любой элемент, являющийся первым в своём родителе

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

- :first-of-type - находит первого потомка своего типа среди детей родителя, то есть первые типы тегов из всех дочерних

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

- :focus - для элементов форм, при фокусе
- :focus-visible
- :focus-within - элементу с фокусом или элементу с потомком, на котором фокус
- :fullscreen - соответствует элементу, который в данный момент находится в полноэкранном режиме, (нет в safari)
- :future - для титров
- :has() - если хотя бы один из относительных селекторов, переданных в качестве аргумента, соответствует хотя бы одному элементу.

```scss
/* Отступ снизу будет обнулён только для тегов <h1>,
следом за которыми идёт тег <p> */
h1:has(+ p) {
  margin-bottom: 0;
}
```

- :host, host-context() - shadow dom
- :hover - активизируется когда курсор мыши находится в пределах элемента, но щелчка по нему не происходит
- :indeterminate - для элементов, которые находятся в неопределенном состоянии (элементы формы),
- :in-range - для инпутов, если значение находится в заданном промежутке,

```html
<form action="" id="form1">
    <ul>Приминаются значения между 1 и 10.
        <li>
            <input id="value1" name="value1" type="number" placeholder="1 to 10" min="1" max="10" value="12">
            <label for="value1">Ваше значение </label>
        </li>
</form>
```

```scss
input:in-range {
  background-color: rgba(0, 255, 0, 0.25);
}
input:out-of-range {
  background-color: rgba(255, 0, 0, 0.25);
  border: 2px solid red;
}
input:in-range + label::after {
  content: " НОРМАЛЬНОЕ";
}
input:out-of-range + label::after {
  content: "вне диапазона!";
}
```

- :invalid - для форм,
- :is() - любой селектор из списка совместим с :matches(), :any()

```scss
// Выбирает какой-либо абзац в шапке, основной части или подвале, который зависал
:is(header, main, footer) p:hover {
  color: red;
  cursor: pointer;
}
// пример с упрощением вложенности
/* Уровень 0 */
h1 {
  font-size: 30px;
}
/* Уровень 1 */
:is(section, article, aside, nav) h1 {
  font-size: 25px;
}
/* Уровень 2 */
:is(section, article, aside, nav) :is(section, article, aside, nav) h1 {
  font-size: 20px;
}
/* Уровень 3 */
:is(section, article, aside, nav)
  :is(section, article, aside, nav)
  :is(section, article, aside, nav)
  h1 {
  font-size: 15px;
}
```

Отличает от where - у where специфичность === 0

- :lang(en|) - элемент:lang(язык) на элементах должен быть определен атрибут lang
- :last-child - если является последним ребенком,

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

- :last-of-type - выберет последний тег,

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

- :left - используется с @-правилом @page, предоставляет все левые страницы печатного документа,
- :link применяется к не посещённым ссылкам a { } и a:link { } по своему результату одинаковые ,
- :local-link - ссылки которые относятся к тому же документу
- :modal - для выбора контента в диалоговом окне

```scss
:modal {
  background-color: beige;
  border: 2px solid burlywood;
  border-radius: 5px;
}
```

```html
<button id="showNumber">Show me</button>

<dialog id="favDialog">
  <form method="dialog">
    <!-- стили применятся к этому контенту -->
    <p>Lucky number is: <strong id="number"></strong></p>
    <button>Close dialog</button>
  </form>
</dialog>
```

- :muted - для видео и аудио, у которых выключен звук
- :not() - принимает селектор,
- :nth-child() - находит один или более элементов, основываясь на их позиции среди группы соседних элементов,

Значения аргумента - odd, even, формула An + B, где n - позиция начиная с 1

- :nth-last-child() - an+b-1 элемент,
- :nth-last-of-type() - последний элемент с заданным тегом
- :nth-of-type() - находит один или более элементов с заданным тегом,
- :only-child - дял элемента, который является единственным предком
- :only-of-type - выбирает такой элемент, который является единственным потомком такого типа,
- :optional - у которых не установлен атрибут required (формы),
- :past - для титров Соответствует элементам перед текущим элементом.
  :placeholder-shown - состояние кода отображается placeholder
- :playing, :paused - для элементов с возможностью воспроизведения
- :out-of-range - противоположность :in-range,
- :read-only - находит элементы, недоступные для редактирования пользователем,
- :read-write - находит элементы, доступные для редактирования пользователем, такие как текстовые поля,
- :required - для форм,
- :right - используется с @-правилом @page,
- :root - элемент, который является корнем документа используется для определения переменных то есть тег html,
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

- :state() - для кастомных элементов
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

- :target-within если элемент является target или включает в себя 'ktvtyn'
- :user-invalid - дял стилизации валидируемых в поле полей, которые имеют type, required
- :user-valid
- :valid - контент которых валиден, в соответствии с типом поля (формы)
- :visited - псевдокласс применяется к посещенным ссылкам
- :where() - :is(), :any()

!!!L-V-H-A-порядок: :link — :visited — :hover — :active дял стилизации ссылок

<!-- BP -------------------------------------------------------------------------------------------------------------------------------------->

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

# Псевдоэлементы

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

# Специфичность CSS-селекторов

- тег и псевдоэлемент имеют специфичность 0001
- класс, псевдокласс, атрибут - 0010
- id имеет специфичность 0100
- inline стиль имеет приоритет 1000

<!-- Вложенность ----------------------------------------------------------------------------------------------------------------------------->

# Вложенность

Позволяет описывать правила внутри других правил. Разница с препроцессорами - не компилируется, а считывается браузером. Специфичность === :is()

```scss
.parent-rule {
  .child-rule {
  }
}

// равнозначные записи
.parent-rule {
}

.parent-rule .child-rule {
}
```

С псевдо классами если не добавить амперсанд

```scss
.parent-rule {
  :hover {
  }
}

.parent-rule {
}

.parent-rule *:hover {
}
```

Использование &:

- При объединении селекторов, например, с помощью составных селекторов или псевдоклассов .
- Для обратной совместимости.
- В качестве визуального индикатора

!!!НЕ предусмотрена конкатенация

```scss
.card {
  .featured & {
  }
}
// равнозначные записи
.card {
}

.featured .card {
}
```

```scss
.card {
  .featured & & & {
  }
}

.card {
}

.featured .card .card .card {
}
```

Если использовать амперсанд наверху - то будет относится к внешнему контексту

Вложенности также подчиняются и @-правила

Поддерживает комбинаторы

```scss
h2 {
  color: tomato;
  + p {
    color: white;
    background-color: black;
  }
}

h2 {
  color: tomato;
  & + p {
    color: white;
    background-color: black;
  }
}
.a {
  /* styles for element with class="a" */
  .b {
    /* styles for element with class="b" which is a descendant of class="a" */
  }
  &.b {
    /* styles for element with class="a b" */
  }
}

.foo {
  /* .foo styles */
  .bar & {
    /* .bar .foo styles */
  }
}
```

Можно вкладывать и медиа выражения
