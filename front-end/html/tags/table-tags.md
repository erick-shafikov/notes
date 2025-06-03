<!-- caption --------------------------------------------------------------------------------------------------------->

# caption

заголовок таблицы. Заголовок помещают сразу после тега table. он всегда должен быть первым вложенным элементом.

Атрибуты:

- Изменения расположения с помощью CSS: caption-side, text-align (устаревшие)
- глобальные

Допустимые родители - table

Когда элемент table, содержащий caption является единственным потомком элемента figure, вам следует использовать элемент figcaption вместо caption

Атрибуты:

- align: left | top | right | bottom

```html
<table>
  <caption>
    Заголовок таблицы
  </caption>
  <tr>
    <th></th>
    <th></th>
  </tr>
</table>
```

<!-- col colgroup --------------------------------------------------------------------------------------------------------->

# col и colgroup

используются для общей семантики колонок. col находится внутри colgroup. Если определено несколько colgroup без col то поведение такое-же как и несколько col

colgroup - создает группу из col, по порядку к каждой из колонок будет применены стили
colgroup долен быть после caption, но до thead, tbody, tfoot

Атрибуты col:

- span - сколько столбцов будет стилизовано
- align (устаревший, в CSS text-align): left | center | right | justify
- bgcolor
- char
- charoff

```html
<table>
  <colgroup>
    <!-- пустой тег col не применит стили к первому столбцу-->
    <col />
    <!-- во втором столбце будет применен background-color-->
    <col style="background-color: yellow" />
  </colgroup>
  <tr>
    <th>Data 1</th>
    <th>Data 2</th>
  </tr>
  <tr>
    <td>Calcutta</td>
    <td>Orange</td>
  </tr>
  <tr>
    <td>Robots</td>
    <td>Jazz</td>
  </tr>
</table>
```

```html
<table>
  <caption>
    Superheros and sidekicks
  </caption>
  <colgroup>
    <col />
    <!-- для ячеек первой колонки класс batman, лоя второй flash-->
    <col span="2" class="batman" />
    <col span="2" class="flash" />
  </colgroup>
  <tr>
    <td></td>
    <th scope="col">Batman</th>
    <th scope="col">Robin</th>
    <th scope="col">The Flash</th>
    <th scope="col">Kid Flash</th>
  </tr>
  <tr>
    <th scope="row">Skill</th>
    <td>Smarts, strong</td>
    <td>Dex, acrobat</td>
    <td>Super speed</td>
    <td>Super speed</td>
  </tr>
</table>
```

Пример только с colgroup

```html
<table>
  <caption>
    Personal weekly activities
  </caption>
  <colgroup span="5" class="weekdays"></colgroup>
  <colgroup span="2" class="weekend"></colgroup>
  <tr>
    <th>Mon</th>
    <th>Tue</th>
    <th>Wed</th>
    <th>Thu</th>
    <th>Fri</th>
    <th>Sat</th>
    <th>Sun</th>
  </tr>
  <tr>
    <td>Clean room</td>
    <td>Football training</td>
    <td>Dance Course</td>
    <td>History Class</td>
    <td>Buy drinks</td>
    <td>Study hour</td>
    <td>Free time</td>
  </tr>
  <tr>
    <td>Yoga</td>
    <td>Chess Club</td>
    <td>Meet friends</td>
    <td>Gymnastics</td>
    <td>Birthday party</td>
    <td>Fishing trip</td>
    <td>Free time</td>
  </tr>
</table>
```

<!-- table --------------------------------------------------------------------------------------------------------->

# table (block)

table - контейнер для таблицы

td - ячейка, table data
th - для создания одной ячейки таблицы, которая будет обозначена как заглавная, шрифт – жирный, выравнивание по центру
tr контейнер для создания строки таблицы

```html
<table _border="1" width="100%" cellpadding="5">
  <!-- отдельный ряд для заголовков -->
  <tr>
    <th>Ячейка 1</th>
    <!-- жирный шрифт, выравнивание по центру -->
    <th>Ячейка 2</th>
    <!-- жирный шрифт, выравнивание по центру -->
  </tr>
   
  <tr>
       
    <td>Ячейка 3</td>
    <!-- обычное форматирование -->
    <td>Ячейка 4</td>
    <!-- обычное форматирование -->
  </tr>
</table>
```

Устаревшие атрибуты table, которые применяются во всех элементах таблицы:

- align - задает выравнивание по краю окна браузера, допустимые значения left, center, right (лучше использовать css)
- bgcolor – цвет заливки (лучше использовать css)
- border – толщина границы в пикселях (лучше использовать css)
- cellpading - определяет расстояние между границей ячейки и ее содержимым, добавляет пустое пространство к ячейке
- cellspacing – задает расстояние между внешними границами ячеек, border принимается расчет
- cols – указывает количество столбцов, помогая загрузки таблицы
- height – высота яичек, при размере меньше, чем факт браузер выставит самостоятельно
- объединение ячеек:
- - rowspan – Объединение ячеек по вертикали
- - colspan – устанавливает число ячеек, которые должны быть объединены по горизонтали
- rules – отображение границ между ячейками, значения (лучше использовать css):
- - cols (между колонами)
- - rows (строками)
- - group, которые определяются наличием тегов thead tfoot tbody colgroup col толщина границы задается с помощью атрибута border
- scope - добавляется к элементу th, сообщает скринридеру какие ячейки являются заголовками, принимает значения:
- - col
- - row
- - colgroup
- - rowgroup
- width – задает ширину таблицы
- id и header позволяют установить взаимодействие между заголовком и ячейками:
- - id - устанавливаем для каждого th
- - headers - для каждого td элемента, в качестве значения строка с id всех заголовков, к которым относится данный header
- summary

Таблица разделяется на 6 слоев - таблица, группы колонок, колонки, группы рядов, ряды, ячейки

<!-- tbody --------------------------------------------------------------------------------------------------------->

# tbody

tbody - неявно встраивается во все таблицы (если его нет, но указать стиль в css, то стили добавятся), можно использовать несколько, если таблица большая

- после любых элементов caption, colgroup, и thead
- может быть более одного
<!-- td --------------------------------------------------------------------------------------------------------->

# td

Может использоваться только внутри table

Атрибуты tr:

- abbr - краткое описание
- axis - список id
- colspan - сколько столбцов нужно объединить, значения выше 1000 - игнорируются
- headers - список строк, каждая из которых соответствует id элементов th, список, если где-то использовался rowspan
- rowspan - объединение рядов, не выше 65534

```html
<thead>
  <tr>
    <!-- колонка purchase -->
    <th id="purchase">Purchase</th>
    <th id="location">Location</th>
    <th id="date">Date</th>
    <th id="evaluation">Evaluation</th>
    <th id="cost">Cost (€)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <!-- ряд haircut -->
    <th id="haircut">Haircut</th>
    <!-- первый элемент таблица соответствует заголовкам location и haircut -->
    <td headers="location haircut">Hairdresser</td>
    <td headers="date haircut">12/09</td>
    <td headers="evaluation haircut">Great idea</td>
    <td headers="cost haircut">30</td>
  </tr>

  ...
</tbody>
```

<!-- tfoot --------------------------------------------------------------------------------------------------------->

# tfoot

Используется для улучшения доступности, при печати документа. Используются вместе с tr и colspan. Встраивается в конце, после tbody. последняя строка таблицы

<!-- th --------------------------------------------------------------------------------------------------------->

# th

Может быть использован только внутри элемента tr

Атрибуты:

- abbr - краткое описание содержимого
- colspan - сколько колонок объединяем далее
- headers - список id тегов th (если невозмоЖно использовать scope)
- rowspan - сколько рядов объединяем далее
- scope - атрибут который нужен в сложных таблицах
- - row - если это заголовок строки,
- - col - если это заголовок ряда. rowgroup, colgroup
- - rowgroup - заголовок принадлежит группе строк и относится ко всем ее ячейкам;
- - colgroup - заголовок принадлежит к colgroup и относится ко всем ее ячейкам.

Альтернатива id и headers

```html
<thead>
  <tr>
    <th id="purchase">Purchase</th>
    <th id="location">Location</th>
    <th id="date">Date</th>
    <th id="evaluation">Evaluation</th>
    <th id="cost">Cost (€)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th id="haircut">Haircut</th>
    <td headers="location haircut">Hairdresser</td>
    <td headers="date haircut">12/09</td>
    <td headers="evaluation haircut">Great idea</td>
    <td headers="cost haircut">30</td>
  </tr>

  ...
</tbody>
```

```html
<table>
  <tr>
    <th scope="col" rowspan="2">Symbol</th>
    <th scope="col" rowspan="2">Code word</th>
    <th scope="col" colspan="2" id="p" headers="i r">Pronunciation</th>
  </tr>
  <tr>
    <th scope="col" id="i" headers="p">IPA</th>
    <th scope="col" id="r" headers="p">Respelling</th>
  </tr>
  <tr>
    <th scope="row">A</th>
    <td>Alfa</td>
    <td>ˈælfa</td>
    <td>AL fah</td>
  </tr>
  <tr>
    <th scope="row">B</th>
    <td>Bravo</td>
    <td>ˈbraːˈvo</td>
    <td>BRAH voh</td>
  </tr>
  <tr>
    <th scope="row">C</th>
    <td>Charlie</td>
    <td>ˈtʃɑːli</td>
    <td>CHAR lee</td>
  </tr>
  <tr>
    <th scope="row">D</th>
    <td>Delta</td>
    <td>ˈdeltɑ</td>
    <td>DELL tah</td>
  </tr>
</table>
```

<!-- thead --------------------------------------------------------------------------------------------------------->

# thead

Объединяет ряд таблицы, где идут заголовки колонок
Находится после caption, colgroup

- thead - первая строка, должен быть после col, colgroup
- тег должен быть до tbody и не входить в него

```html
<table>
  <caption>
    ...
  </caption>
  <thead>
    <tr>
      <th scope="col">Items</th>
      <th scope="col">Expenditure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Donuts</th>
      <td>3,000</td>
    </tr>
    <tr>
      <th scope="row">Stationery</th>
      <td>18,000</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <th scope="row">Totals</th>
      <td>21,000</td>
    </tr>
  </tfoot>
</table>
```

<!-- вложенные таблицы --------------------------------------------------------------------------------------------------------->

# вложенные таблицы

Таблицы можно встраивать друг в друга

```html
<table
  width="200"
  _bgcolor="###"
  cellspacing="0"
  cellpadding="5"
  _border="1"
  _align="right"
>
  <tr>
    <td>Содержимое таблицы</td>
  </tr>
  <p>…если бы здесь был текст, то он обтекал бы таблицу выше</p>
</table>
```

# colspan, rowspan Объединение ячеек

```html
<!-- Неверное объединение -->
<table _border="1" cellspadding="5" width="100%">
  <!-- //результатом будет таблица 3*2, с пустой ячейкой (3;2) -->
  <tr>
    <!-- //в ряд объединяем 2ячейки -->
    <td colspan="2">Ячейка 1</td>
    <!-- //третья в ряду -->
    <td>Ячейка 2</td>
  </tr>
  <tr>
    <td>Ячейка 3</td>
    <td></td>
    <td>Ячейка 4</td>
    <td>//2 ячейки во втором ряду</td>
  </tr>
</table>
```
