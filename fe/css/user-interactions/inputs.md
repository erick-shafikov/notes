# accent-color

определяет цвета интерфейсов взаимодействия с пользователем

для input type: checkbox, radio, range, progress

```scss
.accent-color {
  accent-color: red;
}
```

- - А именно: input type="checkbox", input type="radio", input type="range", progress

# appearance

Определяет внешний вид для элементов взаимодействия

```scss
.appearance {
  appearance: none; //выключает стилизацию
  appearance: auto; //значение предопределенные ОС
  appearance: button; //auto
  appearance: textfield; //auto
  appearance: searchfield;
  appearance: textarea;
  appearance: push-button;
  appearance: button-bevel;
  appearance: slider-horizontal;
  appearance: checkbox;
  appearance: radio;
  appearance: square-button;
  appearance: menulist;
  appearance: menulist-button;
  appearance: listbox;
  appearance: meter;
  appearance: progress-bar;

  /* Частичный список доступных значений в Gecko */
  -moz-appearance: scrollbarbutton-up;
  -moz-appearance: button-bevel;
}
```

# field-sizing (-ff -s)

Размер элемента по контенту или нет

```scss
.field-sizing {
  field-sizing: content; //размер будет по длине строки ввода
  field-sizing: fixed; //фиксированный
}
```

# caret-color

определяет свойство указателя

```scss
.caret-color {
  caret-color: red; //определенный цвет
  caret-color: auto; //обычно current-color
  caret-color: transparent; //невидимая
}
```

# -moz-orient

Определяет расположение элемента горизонтально, вертикально

```scss
.moz-orient {
  -moz-orient: inline | block | horizontal | vertical;
}
```

# -webkit-text-security

символ на который будет заменен текст

```scss
.webkit-text-security {
  -webkit-text-security: circle | disc | square | none;
}
```

# -moz-user-input (-)

Запрет на ввод в поле ввода

```scss
.moz-user-input {
  -moz-user-input: auto;
  -moz-user-input: none;
}
```

<!-- элементы форм и взвимодействия с пользователем ------------------------------------------------------------------------------------------>

# элементы форм и взвимодействия с пользователем

## :active

при клике на элемент может быть как ссылка так и формы, поля формы. В момент между нажатием и отжатием элемента

```scss
a:link {
  color: blue;
}
/* Посещённые ссылки */
a:visited {
  color: purple;
}
/* Ссылки при наведении */
a:hover {
  background: yellow;
}
/* Активные ссылки */
a:active {
  color: red;
}
/* Активные абзацы */
p:active {
  background: #eee;
}
```

## :autofill

если значение в input взято из автозаполнения

## :blank - для пустого поля ввода или для элемента без потомков

## :checked

для input type="radio" или option внутри select,

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

## :default

находит элемент формы, установленный по умолчанию, с атрибутами checked

## :disabled

находит любой отключённый элемент с атрибутом disabled

## :focus

для элементов форм, при фокусе. Никогда не удаляйте фокусный outline (видимый индикатор фокуса), не заменяя его фокусным контуром подходящим под требования

## :enabled

находит любой включённый элемент,

## :indeterminate

для элементов, которые находятся в неопределенном состоянии (элементы формы),

## :in-range

для инпутов, если значение находится в заданном промежутке,

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

## :indeterminate

для полей ввода, которые находятся в неопределенном состоянии

для чекбоксов если значение true через JavaScript

## :invalid

для полей форм с неверной валидацией

## :modal

для выбора контента в диалоговом окне

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

## :optional

у которых не установлен атрибут required (формы)

## :out-of-range

противоположность :in-range

## :placeholder-shown

состояние кода отображается placeholder

## :popover-open

для стилизации элемента popover (с атрибутом popover)

```scss
[popover] {
  position: fixed;
  inset: 0;
  width: fit-content;
  height: fit-content;
  margin: auto;
  border: solid;
  padding: 0.25em;
  overflow: auto;
  color: CanvasText;
  background-color: Canvas;
}
```

## :user-invalid

дял стилизации валидируемых в поле полей, которые имеют type, required

## :user-valid

дял стилизации валидируемых в поле полей, которые имеют type, required

## :valid

дял стилизации валидируемых в поле поле

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

## ::placeholder (-s)

для input текста placeholder,

## ::spelling-error (-ff)

## ::target-text

для прокрученного текста
