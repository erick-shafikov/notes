<!-- abbr ----------------------------------------------------------------------------------------------------------------------->

# abbr (inline)

для аббревиатур,

- атрибут title покажет полную расшифровку
- только глобальные аттрибуты

Может быть использован в паре с тегом [dfn](#dfn-str)

```html
<p>
  <dfn id="html"><abbr title="HyperText Markup Language">HTML</abbr> </dfn> –
  язык разметки...
</p>

<p>
  A <dfn id="spec">Specification</dfn> (<abbr title="Specification">spec</abbr>)
  – документ...
</p>
```

Стал заменой тегу acronym

<!-- address ------------------------------------------------------------------------------------------------------------------->

# address (block)

задает контакты для ближайшего родительского article или body. Для контактной информации использую тег p,

- нельзя вкладывать time, h1-h6.
- обычно внутри footer
- стиль тайкой же i em

<!-- b ----------------------------------------------------------------------------------------------------------------------->

# b (inline)

Жирный шрифт, по важности уступает strong, лучше использовать font-weight. этот тег как и u,i - старые теги использовать если нет более подходящего. Не имеет какой-либо особой важности

Атрибуты:

- глобальные

<!-- bdi --------------------------------------------------------------------------------------------------------------------->

# bdi, bdo (block, HTML5)

bdi Изолирует двунаправленное определение текста

```html
<p dir="ltr">
  This arabic word <bdi>ARABIC_PLACEHOLDER</bdi> is automatically displayed
  right-to-left.
</p>
```

bdo поддерживает атрибут dir ltr и rtl

<!-- blockquote ----------------------------------------------------------------------------------------------------------------->

# blockquote (block)

Если идет цитируемость с другого источника и нужно заключить в новый абзац, если нужна строчная цитата [цитаты строчная](#q)

Будет добавлен абзац

```html
<blockquote cite="href-откуда-цитата">
  <p>текст цитаты</p>
</blockquote>
```

```html
<p>Сверху обычный абзац, а ниже цитата:</p>
<blockquote
  cite="https://developer.mozilla.org/ru/docs/Web/HTML/Element/blockquote"
>
  <p>
    <strong>HTML-элемент <code>&lt;blockquote&gt;</code></strong> (от англ.
    <em>HTML Block Quotation Element</em>) указывает на то, что заключённый в
    нём текст является развёрнутой цитатой.
  </p>
</blockquote>
```

<!-- br --------------------------------------------------------------------------------------------------------------------->

# br

создает разрыв строки, нет стилей, но можно использовать margin, а лучше line-height

<!-- cite  ----------------------------------------------------------------------------------------------->

# cite

ссылка на какой-либо источник

```html
<p>
  Как указано в статье о
  <a href="/ru/docs/Web/HTML/Element/blockquote">
    <cite>блочных цитатах</cite>
  </a>
  :
</p>

<blockquote cite="/ru/docs/Web/HTML/Element/blockquote">
  <p>
    <strong>HTML-элемент <code>&lt;blockquote&gt;</code></strong> (от англ.
    <em>HTML Block Quotation Element</em>) указывает на то, что заключённый в
    нем текст является развёрнутой цитатой.
  </p>
</blockquote>

<p>
  Элемент цитирования <code>&lt;q&gt;</code>
  <q cite="https://developer.mozilla.org/ru/docs/Web/HTML/Element/q">
    предназначен для коротких цитат, не требующих прерывания абзаца
  </q>
  (<a href="/ru/docs/Web/HTML/Element/q"> <cite>Строчные цитаты</cite> </a>).
</p>
```

<!-- code, pre, var, kbd, samp  ----------------------------------------------------------------------------------------------->

# code, pre, var, kbd, samp (block)

- code - для кода
- pre - для кода, в котором ненужно игнорировать пробелы, в него нужно обернуть code, если более одной строки кода в ставке или текст в котором нужно соблюсти пробелы и разрывы строк
- var - для маркировки переменных
- kbd - для маркировки ввода с клавиатуры
- samp - для маркировки ввод кода

```html
<pre><code>const para = document.querySelector('p');

para.onclick = function() {
  alert('Клик по абзацу!');
}</code></pre>

<p>
  Не следует использовать HTML-элементы только для изменения внешнего вида
  текста, такие как <code>&lt;font&gt;</code> и <code>&lt;center&gt;</code>.
</p>

<p>
  В представленном выше примере JavaScript-кода, <var>para</var> представляет
  элемент абзаца.
</p>

<p>
  Выбрать весь текст можно с помощью комбинации клавиш
  <kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>A</kbd>.
</p>

<pre>$ <kbd>ping mozilla.org</kbd>
<samp>PING mozilla.org (63.245.215.20): 56 data bytes
64 bytes from 63.245.215.20: icmp_seq=0 ttl=40 time=158.233 ms</samp></pre>
```

<!-- del ins ---------------------------------------------------------------------------------------------------------------------->

# del

del Отобразит перечеркнутый текст

Атрибуты:

- cite - урл причины удаления
- datetime - дата удаления

```html
<blockquote>
  There is <del>nothing</del> <ins>no code</ins> either good or bad, but
  <del>thinking</del> <ins>running it</ins> makes it so.
</blockquote>
```

<!-- dfn ----------------------------------------------------------------------------------------------------------------->

# dfn (str)

используется в паре с [abbr](#abbr-inline)

```html
<p>
  A <dfn id="def-validator">validator</dfn> is a program that checks for syntax
  errors in code or documents.
</p>
```

использование с abbr

```html
<p>
  <dfn><abbr title="Hubble Space Telescope">HST</abbr></dfn> является одним из
  самых производительных научных инструментов, когда-либо созданных. Он
  находится на орбите более 20 лет, просматривая небо и отправляя данные и
  фотографии беспрецедентного качества и детализации.
</p>

<p>
  Действительно, HST, возможно,
  <!-- на странице будет HST  -->
  <abbr title="Hubble Space Telescope"></abbr> сделал больше для развития науки,
  чем любое другое устройство, когда-либо созданное.
</p>
```

В документе могут быть ссылки-якоря на dfn элемент по id

<!-- em ---------------------------------------------------------------------------------------------------------------------->

# em (str)

Отмечает акцентированный текст, отображается курсивом. Разница с i. em - ударение на содержании, i - слово которое отдичается

<!-- h1-h6 ---------------------------------------------------------------------------------------------------------------------->

# h1-h6 (block)

Один h1 на страниц. h1 === 2rem (16px)

```html
<span style="font-size: 32px; margin: 21px 0;"
  >Это заголовок верхнего уровня?</span
>
```

<!-- i ---------------------------------------------------------------------------------------------------------------------->

# i

Курсив - идиоматический текст, технические термины, таксономические обозначения и т. д.

- Альтернативный голос или настроение
- Таксономические обозначения (например, род и вид « Homo sapiens »)
- Идиоматические термины из другого языка (например, « et cetera »); они должны включать langатрибут, идентифицирующий язык.
- Технические термины
- Транслитерации
- Мысли (например, «Она задавалась вопросом: « О чем вообще говорит этот писатель? »)
- Названия кораблей или судов в западных системах письма (например, «Они искали в доках «Императрицу Галактики» , корабль, на который они были назначены»).

em - обозначения акцента.
strong - для обозначения важности, серьезности или срочности.
mark - для указания релевантности.
cite - для разметки названия произведения, например книги, пьесы или песни.
dfn - для разметки определяющего экземпляра термина.

<!-- ins ---------------------------------------------------------------------------------------------------------------------->

# ins

Обозначает добавленный текст к статье

<!-- mark ----------------------------------------------------------------------------------------------------------------------->

# mark (HTML5)

Предназначен для выделение текста в результате поиска. Отличие от strong - mark используется для отношения к другому контексту

<!-- marquee ----------------------------------------------------------------------------------------------------------------------->

# marquee

для вращающегося текста

<!-- p ------------------------------------------------------------------------------------------------------------------->

# p (block)

HTML не устанавливает перенос текста

абзац – необязательный тег.
Выравнивание с помощью атрибута align, также выравнивать можно с помощью тега div. align может принимать:

- left – выравнивание по левому краю (по умолчанию)
- right – выравнивание по правому краю
- center – по центру
- justify –выравнивание по ширине, для текста длина которого более чем одна строка

Имеет вертикальные margin по 16px

```html
<h1 _align="center">заголовок</h1>
<h2 _align="right">автор</h2>
<p _align="justify">текст</p>
```

По умолчанию разделяются одной строкой

<!-- q ------------------------------------------------------------------------------------------------------------------->

# q

строчная цитата, которая не требует новый абзац в отличает от [цитаты в абзаце](#blockquote)

Атрибут:

- cite - источник цитаты

```html
<p>
  Элемент цитирования <code>&lt;q&gt;</code> предназначен
  <q cite="https://developer.mozilla.org/ru/docs/Web/HTML/Element/q">
    для коротких цитат, не требующих прерывания абзаца
  </q>
  .
</p>
```

<!-- ruby ------------------------------------------------------------------------------------------------------------------->

# ruby rb rt

для аннотации текста,

<!-- s ------------------------------------------------------------------------------------------------------------------->

# s

для перечеркнутого текста, но для неактуального текста - del, есть устаревший - strike

<!-- small ------------------------------------------------------------------------------------------------------------------->

# small (str)

уменьшает на 1 у.е

<!-- span ------------------------------------------------------------------------------------------------------------------->

# span (str)

для определения строчного элемента внутри документа

<!-- strong ------------------------------------------------------------------------------------------------------------------->

# strong (str)

позволяет выделить текст заключенный в тег. Разница с b - strong для более значимого контента, b - для привлечения внимания. em - на него делается более сильный акцент

<!-- sub sup ------------------------------------------------------------------------------------------------------------------->

# sub sup (str)

для надстрочного и подстрочного индексов

<!-- time  ------------------------------------------------------------------------------------------------------------------->

# time (HTML5)

для отображения времени

```html
<!-- Стандартная дата -->
<time datetime="2020-01-20">20 Января 2020</time>
<!-- Только год и месяц -->
<time datetime="2020-01">Январь 2020</time>
<!-- Только месяц и день -->
<time datetime="01-20">20 Января</time>
<!-- Только время, часы и минуты -->
<time datetime="19:30">19:30</time>
<!-- Также вы можете отобразить секунды и миллисекунды! -->
<time datetime="19:30:01.856">19:30:01.856</time>
<!-- Дата и время -->
<time datetime="2020-01-20T19:30">7.30pm, 20 Января 2020</time>
<!-- Дата и время со смещением по часовому поясу -->
<time datetime="2020-01-20T19:30+01:00"
  >7.30pm, 20 Января 2020, — это 8.30pm во Франции.</time
>
<!-- Вызов номера недели -->
<time datetime="2020-W04">Четвёртая неделя 2020</time>
```

<!-- u ------------------------------------------------------------------------------------------------------------------->

# u

подчеркнутый текст волнистой линией

<!-- wbr ------------------------------------------------------------------------------------------------------------------->

# wbr (HTML5)

для переноса слов

```html
<div id="example-paragraphs">
  <p>Fernstraßenbauprivatfinanzierungsgesetz</p>
  <p>Fernstraßen<wbr />bau<wbr />privat<wbr />finanzierungs<wbr />gesetz</p>
  <p>Fernstraßen&shy;bau&shy;privat&shy;finanzierungs&shy;gesetz</p>
</div>
```