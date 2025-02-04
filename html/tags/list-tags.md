<!-- dl, dt, dd ----------------------------------------------------------------------------------------------------------------->

# dl, dt, dd (block)

Каждый такой список начинается с контейнера dl(description list), куда входит тег dt (description term) создающий термин и тег dd (description definition) задающий определение этого термина. Закрывающий тег dt не обязателен, поскольку следующий тег сообщает о завершении предыдущего элемента. Тем не менее, хорошим стилем является закрывать все теги.

dl Имеет вертикальные по 16px
dd имеет margin-left === 2.5rem

```html
Синтаксис
<dl>
  <dt>Термин 1</dt>
  <dd>Определение термина 1</dd>
  <dd>Еще одно определение термина 1</dd>
  <dt>Термин 2</dt>
  <dd>Определение термина 2</dd>
</dl>
```

<!-- li ----------------------------------------------------------------------------------------------------------------------->

# li (block)

элемент для списков ol ul

Атрибуты:

- value - числовой атрибут порядкового номера если это ol, с которого начнется
- type - заменен на css свойство list-style (a - алф, A - АЛФ, i - рим, I- РИМ, 1 - числа)

  <!-- ol ------------------------------------------------------------------------------------------------------------------->

# ol (block)

Список, в котором порядок имеет значение
имеет верхний и нижний margin по 16px === 1em, padding-left === 1.5 em (40px)

Атрибуты:

- reversed
- start - число с которого начинается нумерация, для начала списка с определенного значения

```html
<ol type="I" start="8"></ol>
```

- type - заменен на css свойство list-style (a - алф, A - АЛФ, i - рим, I- РИМ, 1 - числа)

ordered list

```html
<ol>
  <li>Первый пункт</li>
  <li>Первый пункт</li>
</ol>
```

```html
Атрибут type для выбора типа маркеров:
<!-- Арабские числа  -->
<ol type="1"></ol>
<!-- Прописные буквы  -->
<ol type="A"></ol>
<!-- Строчные буквы  -->
<ol type="a"></ol>
<!-- римские числа в верхнем регистре  -->
<ol type="I"></ol>
<!-- римские числа в нижнем регистре  -->
<ol type="i"></ol>
```

<!-- ul ------------------------------------------------------------------------------------------------------------------->

# ul (block)

имеет верхний и нижний margin по 16px === 1em, padding-left === 1.5 em (40px)

!!!Отступы добавляются автоматически

```html
<!-- Список с маркерами в виде круга  -->
<ul type="disc"></ul>
<!-- Список с маркерами в виде окружностей  -->
<ul type="circle"></ul>
<!-- Список с маркерами в виде квадратов  -->
<ul type="square"></ul>
```
