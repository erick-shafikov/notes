<!-- attr() ---------------------------------------------------------------------------------------------------------------------------->

# attr()

Можно доставать значение элемента

```scss
attr(data-count);
attr(title);

/* С типом */
attr(src url);
attr(data-count number);
attr(data-width px);

/* с фоллбэком */
attr(data-count number, 0);
attr(src url, '');
attr(data-width px, inherit);
attr(data-something, 'default');
```

Пример использования

Добавит слово hello в качестве ::before, достанет из аттрибута

```html
<p data-foo="hello">world</p>
```

```scss
p::before {
  content: attr(data-foo) " ";
}
```

<!-- calc-size()---------------------------------------------------------------------------------------------------------------------------->

# calc-size()

Нет в safari и ff

Позволяет вычислять размеры для ключевых слов auto, fit-content, max-content, content, max-content.

```scss
/* получение значений calc-size(), ничего не делать со значениями*/
calc-size(auto, size)
calc-size(fit-content, size)

/* применять изменения к измеримому объекту */
calc-size(min-content, size + 100px)
calc-size(fit-content, size / 2)

/* с функциями */
calc-size(auto, round(up, size, 50px))
```

```scss
section {
  height: calc-size(calc-size(max-content, size), size + 2rem);
}

//тоже самое что и
:root {
  --intrinsic-size: calc-size(max-content, size);
}

section {
  height: calc-size(var(--intrinsic-size), size + 2rem);
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# calc()

Применение различных операций

```scss
 {
  // 110px
  width: calc(10px + 100px);
  // 10em
  width: calc(2em * 5);
  // в зависимости от ширины
  width: calc(100% - 32px);
  --predefined-width: 100%;
  /* Output width: Depends on the container's width */
  width: calc(var(--predefined-width) - calc(16px * 2));
}
```

<!-- counter()------------------------------------------------------------------------------------------------------------------------------>

# counter() counter-reset() counter-increment()

активирует запуск счетчика на элементах

```scss
.double-list {
  counter-reset: count -1;
}

.double-list li {
  counter-increment: count 2;
}

.double-list li::marker {
  content: counter(count, decimal) ") ";
}
```

создаст список с 1-3-5-7-9

```html
<p>Best Dynamic Duos in Sports:</p>
<ol class="double-list">
  <li>Simone Biles + Jonathan Owens</li>
  <li>Serena Williams + Venus Williams</li>
  <li>Aaron Judge + Giancarlo Stanton</li>
  <li>LeBron James + Dwyane Wade</li>
  <li>Xavi Hernandez + Andres Iniesta</li>
</ol>
```

Пример с глобальным счетчиком

```scss
body {
  // Устанавливает значение счётчика, равным 0 переменная - section
  counter-reset: section;
}

h3::before {
  // Инкриминирует счётчик section = ++section
  counter-increment: section;
  // Отображает текущее значение счётчика counter - достает значение
  content: "Секция " counter(section) ": ";
}
```

```html
<h3>Вступление</h3>
<h3>Основная часть</h3>
<h3>Заключение</h3>
```

Вывод
Секция 1: Вступление
Секция 2: Основная часть
Секция 3: Заключение

<!-- counters()------------------------------------------------------------------------------------------------------------------------------------>

# counters()

для вложенных списков

```scss
ol {
  counter-reset: index;
  list-style-type: none;
}

li::before {
  counter-increment: index;
  content: counters(index, ".", decimal) " ";
}
```

```scss
ol {
  counter-reset: section; /* Создаёт новый счётчик для каждого тега <ol> */
  list-style-type: none;
}

li::before {
  counter-increment: section; /* Инкриминируется только счётчик текущего уровня вложенности */
  // уже не counter
  content: counters(section, ".") " "; /* Добавляем значения всех уровней вложенности, используя разделитель '.' */
  /* Если необходима поддержка < IE8, необходимо убедиться, что после разделителя ('.') не стоит пробел */
}
```

<!-- fit-content()  ------------------------------------------------------------------------------------------------------------------------>

# fit-content()

Позволяет взять размер меньший из максимального и максимального между минимальным и заданным

fit-content(argument) = min(maximum size, max(minimum size, argument)).

<!-- env()---------------------------------------------------------------------------------------------------------------------------------->

# env()

Позволяет получить значение какого-либо свойства предопределенное системой

```scss
body {
  padding: env(safe-area-inset-top, 20px) env(safe-area-inset-right, 20px) env(
      safe-area-inset-bottom,
      20px
    ) env(safe-area-inset-left, 20px);
}
```

<!--  image-set() ---------------------------------------------------------------------------------------------------------------------------->

# image-set()

Позволяет выбрать наиболее подходящее изображение

```scss
.box {
  background-image: url("large-balloons.jpg");
  background-image: image-set(
    "large-balloons.avif" type("image/avif"),
    "large-balloons.jpg" type("image/jpeg")
  );
}
```

<!-- is() ---------------------------------------------------------------------------------------------------------------------------------->

# is()

позволяет проверить на наличие поддержки того или иного свойства

```scss
is(--moz-prefix) {
  //
}
```

<!-- layer() ---------------------------------------------------------------------------------------------------------------------------->

# layer()

для работы с пространствами

```scss
 {
  @import url layer(layer-name);
  @import "dark.css" layer(framework.themes.dark);
}
```

<!-- light-dark() ---------------------------------------------------------------------------------------------------------------------------->

# light-dark()

для определения темы

```scss
 {
  :root {
    color-scheme: light dark;
  }
  body {
    color: light-dark(#333b3c, #efefec);
    background-color: light-dark(#efedea, #223a2c);
  }
}
```

```scss
:root {
  /* this has to be set to switch between light or dark */
  color-scheme: light dark;

  --light-bg: ghostwhite;
  --light-color: darkslategray;
  --light-code: tomato;

  --dark-bg: darkslategray;
  --dark-color: ghostwhite;
  --dark-code: gold;
}
* {
  background-color: light-dark(var(--light-bg), var(--dark-bg));
  color: light-dark(var(--light-color), var(--dark-color));
}
code {
  color: light-dark(var(--light-code), var(--dark-code));
}
```

<!-- matrix() matrix3d() ---------------------------------------------------------------------------------------------------------------------------->

# matrix() matrix3d()

Для трансформации в 2d и 3d

```scss
 {
  //
}
```

<!-- paint() ---------------------------------------------------------------------------------------------------------------------------->

# paint()

Декорация

```scss
li {
  background-image: paint(boxbg);
  --boxColor: hsl(55 90% 60% / 100%);
}
li:nth-of-type(3n) {
  --boxColor: hsl(155 90% 60% / 100%);
  --widthSubtractor: 20;
}
li:nth-of-type(3n + 1) {
  --boxColor: hsl(255 90% 60% / 100%);
  --widthSubtractor: 40;
}
```

<!-- palette-mix()  ---------------------------------------------------------------------------------------------------------------------------->

# palette-mix()

дял создания нового шрифта на основе двух других

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# ray()

Отклонение от оси при создании анимации по clip-path

```scss
/* all parameters specified */
offset-path: ray(50deg closest-corner contain at 100px 20px);

/* two parameters specified, order does not matter */
offset-path: ray(contain 200deg);

/* only one parameter specified */
offset-path: ray(45deg);
```

<!-- repeat()  ----------------------------------------------------------------------------------------------------------------------------->

# repeat()

Функция позволяет задать повторяющиеся величины

```scss
 {
  grid-template-columns: repeat(2, 60px);
}
```

<!-- scroll() (scroll-driven animation) ---------------------------------------------------------------------------------------------------->

# scroll() (scroll-driven animation)

Функция для отслеживания временной шкалы анонимной анимации зависящей от скролла

```scss
 {
  animation-timeline: scroll();

  /* Values for selecting the scroller element */
  animation-timeline: scroll(nearest); /* Default */
  animation-timeline: scroll(root);
  animation-timeline: scroll(self);

  /* Values for selecting the axis */
  animation-timeline: scroll(block); /* Default */
  animation-timeline: scroll(inline);
  animation-timeline: scroll(y);
  animation-timeline: scroll(x);

  /* Examples that specify scroller and axis */
  animation-timeline: scroll(block nearest); /* Default */
  animation-timeline: scroll(inline root);
  animation-timeline: scroll(x self);
}
```

<!-- symbols()------------------------------------------------------------------------------------------------------------------------------>

# symbols()

Позволяет определить стиль счетчика в списках в отличие от @counter-style можно использовать один раз

```scss
ol {
  list-style: symbols(cyclic "*" "†" "‡");
}
```

<!-- url()---------------------------------------------------------------------------------------------------------------------------->

# url()

использование внешних ресурсов

```scss
 {
  background-image: url("star.gif");
  list-style-image: url("../images/bullet.jpg");
  content: url("my-icon.jpg");
  cursor: url(my-cursor.cur);
  border-image-source: url("/media/diamonds.png");
  src: url("fantastic-font.woff");
  offset-path: url(#path);
  mask-image: url("masks.svg#mask1");

  /* Properties with fallbacks */
  cursor: url(pointer.cur), pointer;

  /* Associated short-hand properties */
  background: url("star.gif") bottom right repeat-x blue;
  border-image: url("/media/diamonds.png") 30 fill / 30px / 30px space;

  /* As a parameter in another CSS function */
  background-image: cross-fade(20% url(first.png), url(second.png));
  mask-image: image(
    url(mask.png),
    skyblue,
    linear-gradient(rgb(0 0 0 / 100%), transparent)
  );

  /* as part of a non-shorthand multiple value */
  content: url(star.svg) url(star.svg) url(star.svg) url(star.svg) url(star.svg);

  /* at-rules */
  @document url("https://www.example.com/")
  {
    /* … */
  }
  @import url("https://www.example.com/style.css");
  @namespace url(http://www.w3.org/1999/xhtml);
}
```

<!-- var() ---------------------------------------------------------------------------------------------------------------------------->

# var()

```scss
 {
  .component .header {
    color: var(
      --header-color,
      blue
    ); /* header-color не существует, поэтому используется blue */
  }
}
```

<!-- view() ---------------------------------------------------------------------------------------------------------------------------->

# view()

Функция для отслеживания временной шкалы анонимной анимации зависящей от видимости элемента от скролла

```scss
.view-function {
  /* Function with no parameters set */
  animation-timeline: view();

  /* Values for selecting the axis */
  animation-timeline: view(block); /* Default */
  animation-timeline: view(inline);
  animation-timeline: view(y);
  animation-timeline: view(x);

  /* Values for the inset */
  animation-timeline: view(auto); /* Default */
  animation-timeline: view(20%);
  animation-timeline: view(200px);
  animation-timeline: view(20% 40%);
  animation-timeline: view(20% 200px);
  animation-timeline: view(100px 200px);
  animation-timeline: view(auto 200px);

  /* Examples that specify axis and inset */
  animation-timeline: view(block auto); /* Default */
  animation-timeline: view(inline 20%);
  animation-timeline: view(x 200px auto);
}
```

# ----------------------------------

<!--  Математически операции ---------------------------------------------------------------------------------------------------------------------------->

# математически операции

- abs() - модуль
- acos()
- asin()
- atan()
- atan2()
- cos()
- clamp()
- - Если значение, которое необходимо зафиксировать, меньше переданного минимального значения, функция вернет минимальное значение.
- - Если значение, которое необходимо зафиксировать, больше переданного максимального значения, функция вернет максимальное значение.
- - Если значение, которое необходимо зафиксировать, находится между переданными минимальным и максимальным значениями, функция вернет исходное значение, которое необходимо зафиксировать.

```scss
.clamp {
  width: clamp(10px, 4em, 80px); //
}
```

- exp()
- hypot()
- log()
- max(first value, second value, third value, ...)
- min(first value, second value, third value, ...)
- minmax() - определяет диапазон больший или равный меньшему, но не больше максимального
- mod()
- rem()
- pow()
- round()
- sign()
- sin()
- sqrt()
- tan()

<!-- filter функции ------------------------------------------------------------------------------------------------------------------------>

# фильтры функции

```scss
 {
  filter: brightness();
  filter: contrast();
  filter: drop-shadow(); //для отбрасывания тени
  filter: grayscale(); //оттенок серого
  filter: hue-rotate(); //
  filter: invert();
  filter: opacity(); //прозрачность
  filter: saturate();
  filter: sepia();
}
```

drop-shadow()

Создаст тень

```scss
 {
  filter: drop-shadow(30px 10px 4px #4444dd);
}
```

<!-- геометрические фигуры  ---------------------------------------------------------------------------------------------------------------------------->

# геометрические фигуры

```scss
 {
  // круг
  clip-path: circle(50px);
  clip-path: circle(6rem at right center);
  clip-path: circle(10% at 2rem 90%);
  // для создания эллипса
  clip-path: ellipse();
  // определяет область round 50px - border радиус
  clip-path: inset(45px 50px 15px 0 round 50px);
  // определяет svg
  clip-path: path(
    "M  20  240 \
 L  20  80 L 160  80 \
 L 160  20 L 280 100 \
 L 160 180 L 160 120 \
 L  60 120 L  60 240 Z"
  );
  clip-path: polygon(
    0% 20%,
    60% 20%,
    60% 0%,
    100% 50%,
    60% 100%,
    60% 80%,
    0% 80%
  );
  // прямоугольник
  clip-path: rect(50px 70px 80% 20%);
  //произвольная прямая
  clip-path: xywh(1px 2% 3px 4em round 0 1% 2px 3em);
}
```

<!-- работа с цветами ---------------------------------------------------------------------------------------------------------------------------->

# работа с цветами

- color-contrast()
- color-mix() - для смешивания двух цветов
- color() - Позволяет задавать цветовые пространства

```html
<div data-color="red"></div>
<div data-color="green"></div>
<div data-color="blue"></div>
```

```scss
[data-color="red"] {
  background-color: color(xyz 45 20 0);
}

[data-color="green"] {
  background-color: color(xyz-d50 0.3 80 0.3);
}

[data-color="blue"] {
  background-color: color(xyz-d65 5 0 50);
}
```

- device-cmyk()
- hsl()
- hwb()
- lab()
- lch()
- oklab()
- oklch()

- rgb()

<!-- Градиенты  ---------------------------------------------------------------------------------------------------------------------------->

# градиенты

## linear-gradient()

linear-gradient() - создает линейный градиент

```scss
.simple-linear {
  //двухцветные с плавным переходом
  background: linear-gradient(blue, pink);
  background: linear-gradient(#e66465, #9198e5);
  // по умолчанию будут равно распределены
  background: linear-gradient(red, yellow, blue, orange);
  //распределение в неравных пропорциях
  background: linear-gradient(to left, lime 28px, red 77%, cyan);
  //конкретные промежутки
  background: linear-gradient(
    to left,
    lime 25%,
    red 25% 50%,
    cyan 50% 75%,
    yellow 75%
  );
  background: linear-gradient(to left, #333, #333 50%, #eee 75%, #333 75%);
  // для резкого перехода проценты в сумме должны быть равны 100
  background: linear-gradient(to bottom left, cyan 50%, palegoldenrod 50%);
  // подсказка для перехода 10% займет blue
  background: linear-gradient(blue, 10%, pink);
}

// поменять направление
.horizontal-gradient {
  background: linear-gradient(to right, blue, pink);
  // диагональный
  background: linear-gradient(to bottom right, blue, pink);
  // использование углов
  background: linear-gradient(70deg, blue, pink);
  background: linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c);
}
```

```scss
// можно комбинировать с изображениями
.layered-image {
  background: linear-gradient(to right, transparent, mistyrose),
    url("critters.png");
}
```

Наслоение нескольких друг на друга

```scss
 {
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

## radial-gradient()

```scss
// расположение центра
.radial-gradient {
  background: radial-gradient(at 0% 30%, red 10px, yellow 30%, #1e90ff 50%);
}
//размер определяется расстоянием от начальной точки (центра) до ближайшей стороны блока.
.radial-ellipse-side {
  background: radial-gradient(
    ellipse closest-side,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // устанавливает размер градиента значением расстояния от начальной точки до самого дальнего угла блока.
  background: radial-gradient(
    ellipse farthest-corner at 90% 90%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // по x - 25% от контейнера, по y -75%
  background: radial-gradient(
    circle closest-side at 25% 75%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
}
```

## conic-gradient()

conic-gradient() - создает круговой градиент

```scss
.conic-gradient {
  // смещение центра
  background: conic-gradient(at 0% 30%, red 10%, yellow 30%, #1e90ff 50%);
  // поворот поворота
  background: conic-gradient(
    from 45deg,
    red,
    orange,
    yellow,
    green,
    blue,
    purple
  );
}
```

## repeating-linear-gradient

repeating-linear-gradient() - линии

```scss
.repeating-linear {
  background: repeating-linear-gradient(
    -45deg,
    red,
    red 5px,
    blue 5px,
    blue 10px
  );
}
```

- repeating-conic-gradient() - лучи из центра
- repeating-radial-gradient() - круги из центра

# функции для свойства transform

- perspective() - создание перспективы
- rotate() - вращение
- rotate3d() - вращение
- rotateX() - вращение по горизонтальной оси
- rotateY() - вращение по вертикальной оси
- rotateZ() - по перпендикулярной
- scale() - растягивание
- scale3d() - растягивание
- - scaleX() - растягивание
- - scaleY() - растягивание
- - scaleZ() - растягивание
- skew() - скашивание
- - skewX() - скашивание
- - skewY() - скашивание
- translate() - перемещение по плоскости
- - translate3d()
- - translateX()
- - translateY()
- - translateZ()
