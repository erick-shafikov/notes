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
