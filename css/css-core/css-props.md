<!--  accent-color ------------------------------------------------------------------------------------------------------------------------------------>

# accent-color

определяет цвета интерфейсов взаимодействия с пользователем

```scss
 {
  //
  accent-color: red;
}
```

<!-- align-content (flex) --------------------------------------------------------------------------------------------------------------------->

# align-content (flex)

```scss
.align-content {
  align-content: flex-start;
  align-content: flex-end;
  align-content: center;
  align-content: space-between;
  align-content: space-around;
  align-content: stretch;
}
```

<!-- align-items (flex) ------------------------------------------------------------------------------------------------------------------>

# align-items (flex, grid)

Выравнивание по поперечной оси

```scss
.flex {
  /* высота по умолчанию  */
  align-items: stretch;
  /*по верхнему краю (к верху) */
  align-items: start;
  /*по нижнему краю (к низу) */
  align-items: end;
  /*отцентрирует */
  align-items: center;
  /* выравнивает текст внутри элементов */
  align-items: baseline;
}
```

<!-- align-self ---------------------------------------------------------------------------------------------------------------------------->

# align-self

Выравнивание элемента управляемое самим элементом

```scss
 {
  align-self: center; /* Put the item around the center */
  align-self: start; /* Put the item at the start */
  align-self: end; /* Put the item at the end */
  align-self: self-start; /* Align the item flush at the start */
  align-self: self-end; /* Align the item flush at the end */
  align-self: flex-start; /* Put the flex item at the start */
  align-self: flex-end; /* Put the flex item at the end */

  /* Baseline alignment */
  align-self: baseline;
  align-self: first baseline;
  align-self: last baseline;
  align-self: stretch; /* Stretch 'auto'-sized items to fit the container */
}
```

<!-- anchor-name (якоря) --------------------------------------------------------------------------------------------------------------------->

# anchor-name (якоря)

добавляет имя для якоря, к которому в последствии присоединится элемент

```scss
 {
  //
  anchor-name: --__someAnchorName__;
}
```

<!-- animation ------------------------------------------------------------------------------------------------------------------------------->

# animation

это сокращенная запись для animation-name + animation-duration + animation-timing-function + animation-delay + animation-iteration-count + animation-direction + animation-fill-mode + animation-play-state

```scss
 {
  /* @keyframes duration | timing-function | delay |
   iteration-count | direction | fill-mode | play-state | name */
  animation: 3s ease-in 1s infinite reverse both running slidein;
}
```

<!-- animation-composition  ------------------------------------------------------------------------------------------------------------------------------------>

# animation-composition

Позволяет применять несколько анимации, полезно когда применяем два раза transform etc

```scss
 {
  animation-composition: replace; //будут перезаписываться анимации одного свойства
  animation-composition: add; // add и accumulate применяются по разному
  animation-composition: accumulate;
}
```

```scss
.square {
  height: 100px;
  width: 100px;
  background-color: gold;
  //применяем две анимации move и bounce
  animation: 2s ease-in-out infinite alternate move, 0.3s ease-in-out infinite
      alternate bounce;
  /* по умолчанию */
  /* animation-composition: replace; */

  /* смешаются 2 анимации */
  animation-composition: add;

  /* смешаются 2 анимации как и add */
  /* animation-composition: accumulate; */
}

@keyframes move {
  0% {
    transform: translateX(0);
  }

  100% {
    transform: translateX(calc(100vw - 140px));
  }
}

@keyframes bounce {
  0% {
    transform: translateY(0);
  }
  100% {
    transform: translateY(100px);
  }
}
```

<!--animation-delay--------------------------------------------------------------------------------------------------------------------------->

# animation-delay

Время задержки перед стартом. При указании неправильных значений, не применится

```scss
 {
  //
  animation-delay: 1s; //через секунду
  animation-delay: -1s; //при указании отрицательных значений, анимация будет проигрываться с того времени анимации, которая указана с отрицательным значением
}
```

<!--animation-direction----------------------------------------------------------------------------------------------------------------------->

# animation-direction

```scss
 {
  /* Одиночная анимация */
  animation-direction: normal; //после проигрыша анимации - позиция сбросится
  animation-direction: reverse; //проигрыш задом наперед
  animation-direction: alternate; // в первом цикле normal, во втором reverse
  animation-direction: alternate-reverse; //противоположно alternate

  /* Несколько анимаций */
  animation-direction: normal, reverse;
  animation-direction: alternate, reverse, normal;

  /* Глобальные значения */
  animation-direction: inherit;
  animation-direction: initial;
  animation-direction: unset;
}
```

<!--animation-duration  ---------------------------------------------------------------------------------------------------------------------->

# animation-duration

Продолжительность анимации

```scss
 {
  animation-duration: 1s; //отрицательное и нулевое значение будет проигнорировано
}
```

<!--animation-fill-mode  --------------------------------------------------------------------------------------------------------------------->

# animation-fill-mode

как нужно применять стили к объекту анимации до и после проигрыша

```scss
 {
  /* Ключевые слова */
  animation-fill-mode: none; //стили не будут применены до и после
  animation-fill-mode: forwards; // 100% или to в зависимости
  animation-fill-mode: backwards;
  animation-fill-mode: both;

  /* Несколько значений могут быть заданы через запятую. */
  /* Каждое значение соответствует для анимации в animation-name. */
  animation-fill-mode: none, backwards;
  animation-fill-mode: both, forwards, none;
}
```

<!--animation-iteration-count  --------------------------------------------------------------------------------------------------------------->

# animation-iteration-count

```scss
 {
  animation-iteration-count: infinite; //анимация будет проигрываться бесконечно
  animation-iteration-count: 3; //3 раза
  animation-iteration-count: 2.5; //2 с половиной раза
}
```

<!--animation-name  -------------------------------------------------------------------------------------------------------------------------->

# animation-name

имя анимации

```scss
 {
  animation-name: test_05; //-specific, sliding-vertically
}
```

<!--animation-play-state  -------------------------------------------------------------------------------------------------------------------->

# animation-play-state

Состояние анимации - пауза или проигрыш, если запустить анимацию после паузы она начнется с того места, где остановилась. Позволяет управлять анимацией из скрипта

```scss
 {
  animation-play-state: running; //
  animation-play-state: paused; //
}
```

<!-- animation-range (scroll-driven-animation)---------------------------------------------------------------------------------------------------------------------------->

# animation-range = animation-range-start + animation-range-end

Позволяет определить настройки срабатывания анимации, относительно начала и конце шкалы

```scss
 {
  /* single keyword or length percentage value */
  animation-range: normal; /* Equivalent to normal normal */
  animation-range: 20%; /* Equivalent to 20% normal */
  animation-range: 100px; /* Equivalent to 100px normal */

  /* single named timeline range value */
  animation-range: cover; /* Представляет полный диапазон именованной временной шкалы 0% - начал входить*/
  animation-range: contain; /* элемент полностью входит*/
  animation-range: cover 20%; /* Equivalent to cover 20% cover 100% */
  animation-range: contain 100px; /* Equivalent to contain 100px cover 100% */

  /* two values for range start and end */
  animation-range: normal 25%;
  animation-range: 25% normal;
  animation-range: 25% 50%;
  animation-range: entry exit; /* exit - начал выходить */
  animation-range: cover cover 200px; /* Equivalent to cover 0% cover 200px */
  animation-range: entry 10% exit; /* entry - начал входить */
  animation-range: 10% exit 90%;
  animation-range: entry 10% 90%;
  // entry-crossing - пересек
  // exit-crossing вышел
}
```

<!--animation-timing-function  --------------------------------------------------------------------------------------------------------------->

# animation-timing-function

```scss
 {
  animation-timing-function: ease;
  animation-timing-function: ease-in;
  animation-timing-function: ease-out;
  animation-timing-function: ease-in-out;
  animation-timing-function: linear;
  animation-timing-function: step-start;
  animation-timing-function: step-end;

  // С помощью функций
  animation-timing-function: cubic-bezier(0.1, 0.7, 1, 0.1);
  animation-timing-function: steps(4, end);

  // С помощью функций шагов
  animation-timing-function: steps(4, jump-start);
  animation-timing-function: steps(10, jump-end);
  animation-timing-function: steps(20, jump-none);
  animation-timing-function: steps(5, jump-both);
  animation-timing-function: steps(6, start);
  animation-timing-function: steps(8, end);

  /* Multiple animations */
  animation-timing-function: ease, step-start, cubic-bezier(0.1, 0.7, 1, 0.1);

  /* Global values */
  animation-timing-function: inherit;
  animation-timing-function: initial;
  animation-timing-function: unset;
}
```

<!-- animation-timeline (scroll-driven-animation)-------------------------------------------------------------------------------------------------------------------->

# animation-timeline (scroll-driven-animation)

Следующие типы временных шкал могут быть установлены с помощью animation-timeline:

- ременная шкала документа по умолчанию, со старта открытия страницы
- Временная шкала прогресса прокрутки, в свою очередь они делятся на:
- - Именованная временная шкала прогресса прокрутки заданная с помощью [scroll-timeline](#scroll-timeline--scroll-timeline-name-)
- - анонимная задается с помощью функции scroll()
- Временная шкала прогресса просмотра (видимость элемента) делится на
- - Именованная временная шкала прогресса [view-timeline](#view-timeline)
- - Анонимная временная шкала прогресса просмотра

```scss
 {
  animation-timeline: none;
  animation-timeline: auto;

  /* Single animation named timeline */
  animation-timeline: --timeline_name;

  /* Single animation anonymous scroll progress timeline */
  animation-timeline: scroll();
  animation-timeline: scroll(scroller axis);

  /* Single animation anonymous view progress timeline */
  animation-timeline: view();
  animation-timeline: view(axis inset);

  /* Multiple animations */
  animation-timeline: --progressBarTimeline, --carouselTimeline;
  animation-timeline: none, --slidingTimeline;
}
```

<!-- appearance ---------------------------------------------------------------------------------------------------------------------------->

# appearance

Определяет внешний вид для элементов взаимодействия

```scss
.appearance {
  appearance: none; //выключает стилизацию
  appearance: auto; //значение предопределенные ОС
  appearance: menulist-button; //auto
  appearance: textfield; //auto
  appearance: button;
  appearance: checkbox;
}
```

<!-- aspect-ratio  ------------------------------------------------------------------------------------------------------------------------------------>

# aspect-ratio

позволяет настроить пропорции контейнера

```scss
.aspect-ratio {
  aspect-ratio: 1 / 1;
  aspect-ratio: 1;

  /* fallback to 'auto' for replaced elements */
  aspect-ratio: auto 3/4;
  aspect-ratio: 9/6 auto;
}
```

<!-- backface-visibility  ------------------------------------------------------------------------------------------------------------------>

# backface-visibility

будет видна или нет часть изображения в 3d, которая определена как задняя часть

```scss
.backface-visibility {
  backface-visibility: visible;
  backface-visibility: hidden;
}
```

<!-- backdrop-filter ----------------------------------------------------------------------------------------------------------------------->

# backdrop-filter

позволяет применить фильтр к контенту, который находится поверх контейнера с background-color или background-image

```scss
 {
  backdrop-filter: none;

  /* фильтр URL в SVG */
  backdrop-filter: url(commonfilters.svg#filter);

  /* значения <filter-function> */
  backdrop-filter: blur(2px);
  backdrop-filter: brightness(60%);
  backdrop-filter: contrast(40%);
  backdrop-filter: drop-shadow(4px 4px 10px blue);
  backdrop-filter: grayscale(30%);
  backdrop-filter: hue-rotate(120deg);
  backdrop-filter: invert(70%);
  backdrop-filter: opacity(20%);
  backdrop-filter: sepia(90%);
  backdrop-filter: saturate(80%);

  /* Несколько фильтров */
  backdrop-filter: url(filters.svg#filter) blur(4px) saturate(150%);
}
```

Пример контента с изображением, фон которого будет размыт

```scss
// контент фон которого будет размыт
.box {
  background-color: rgba(255, 255, 255, 0.3);
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
}

// изображение
img {
  background-image: url("anemones.jpg");
  background-position: center center;
  background-repeat: no-repeat;
  background-size: cover;
}

.container {
  align-items: center;
  display: flex;
  justify-content: center;
  height: 100%;
  width: 100%;
}
```

<!-- background ------------------------------------------------------------------------------------------------------------------->

# background

это короткая запись для background-attachment + background-clip + background-color + background-image + background-origin + background-position + background-repeat + background-size

```scss
 {
  //два цвета смешаются ровно посередине
  background: linear-gradient(#e66465, #9198e5);
  // множественный градиент
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

Сокращенная запись

```scss
.box {
  background: linear-gradient(
        105deg,
        rgb(255 255 255 / 20%) 39%,
        rgb(51 56 57 / 100%) 96%
      ) center center / 400px 200px no-repeat, url(big-star.png) center
      no-repeat, rebeccapurple;
}
```

<!--background-attachment ------------------------------------------------------------------------------------------------------------------>

# background-attachment

Определяет поведения заднего фона при прокрутке

```scss
 {
  background-attachment: scroll; //изображение позади будет прокручиваться
  background-attachment: fixed; //изображение позади не будет прокручиваться
  background-attachment: local; //в зависимости от прокручивания контента позади которого будет изображение
}
```

<!-- background-blend-mode ----------------------------------------------------------------------------------------------------------------->

# background-blend-mode

Определяет как будут смешиваться наслаиваемые цвета и изображения

```scss
 {
  background-blend-mode: darken | luminosity...;
}
```

<!--background-clip  ----------------------------------------------------------------------------------------------------------------------->

# background-clip

Настраивает как будет обрезаться изображение, которое находится позади

```scss
 {
  background-clip: border-box; //до края границы
  background-clip: padding-box; // до края отступа
  background-clip: content-box; // внутри содержимого
  background-clip: text; //обрезка текстом
}
```

<!--background-image  ---------------------------------------------------------------------------------------------------------------------->

# background-image

Может быть как градиентом так и изображением

```scss
 {
  background-image: linear-gradient(black, white);
  background-image: url("image.png");

  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png); // несколько изображений

  //Создание изображения с наложением градиента
  background-image: linear-gradient(
      rgba($color-secondary, 0.93),
      rgba($color-secondary, 0.93)
    ), url("../img/hero.jpeg");
  //пример с распределением градиента
  background-image: linear-gradient(
      105deg,
      rgba($color-white, 0.9) 0%,
      rgba($color-white, 0.9) 50%,
      rgba($color-white, 0.9),
      transparent 50%
    ), url("../img/nat-10.jpg");
}
```

<!--background-origin  --------------------------------------------------------------------------------------------------------------------->

# background-origin

как расположить изображение относительно рамок и контента

```scss
 {
  background-repeat: no-repeat; // сначала нужно отключить повтор изображения
  //позиционирование
  background-origin: border-box; //растянуть по всему контейнеру, Фон располагается относительно рамки.
  background-origin: padding-box; //не включая рамки, Фон расположен относительно поля отступа.
  background-origin: content-box; //только по границам контента, Фон располагается относительно поля содержимого.
}
```

<!--background-position--------------------------------------------------------------------------------------------------------------------->

# background-position

```scss
 {
  // прилепить к краям
  background-position: top;
  background-position: bottom;
  background-position: left;
  background-position: right;
  background-position: center;
  // сдвиг в процентах и единицах
  background-position: 25% 75%;
  background-position: 0 0;
  background-position: 1cm 2cm;
  background-position: 10ch 8em;
  // точное расположение относительно краев
  background-position: bottom 10px right 20px;
  background-position: right 3em bottom 10px;
  background-position: bottom 10px right;
  background-position: top right 10px;
  // для нескольких изображений
  background-position: 0 0, center;

  // несколько изображений
  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png);
  background-repeat: no-repeat, repeat-x, repeat; // для image1.png будет применено no-repeat так как свойства применяются циклично
}
```

<!--background-position-x  ----------------------------------------------------------------------------------------------------------------->

# background-position-x и background-position-y

определяет горизонтальную позицию изображения

```scss
 {
  background-position-x: left;
  background-position-x: center;
  background-position-x: right;

  /* <percentage> values */
  background-position-x: 25%;

  /* <length> values */
  background-position-x: 0px;
  background-position-x: 1cm;
  background-position-x: 8em;

  /* Side-relative values */
  background-position-x: right 3px;
  background-position-x: left 25%;

  /* Multiple values */
  background-position-x: 0px, center;
}
```

<!--background-repeat----------------------------------------------------------------------------------------------------------------------->

# background-repeat

```scss
 {
  background-repeat: repeat; // по умолчанию повтор включен
  background-repeat: repeat-x; // repeat no-repeat
  background-repeat: repeat-y; // no-repeat repeat
  background-repeat: space; // будет заполнено не обрезая изображения
  background-repeat: round; //
  background-repeat: no-repeat; // отключить повтор

  // варианты повтора изображения можно задавать для вертикальной и горизонтальной осей
  background-repeat: repeat space;
  background-repeat: repeat repeat;
  background-repeat: round space;
  background-repeat: no-repeat round;
}
```

<!--background-size ------------------------------------------------------------------------------------------------------------------------>

# background-size

Управление размером изображения

```scss
 {
  background-size: cover; // cover - растянет изображение по всему блоку сохраняя пропорции, но обрежет при надобности
  background-size: contain; //  contain - растянет по всем блоку но изменит пропорции

  /* Указано одно значение - ширина изображения, */
  /* высота в таком случае устанавливается в auto */
  background-size: 50%;
  background-size: 3em;
  background-size: 12px;
  background-size: auto; // растягивает сохраняя пропорции

  // два значения - по горизонтали и вертикали
  background-size: 50% auto;
  background-size: 3em 25%;
  background-size: auto 6px;
  background-size: auto auto;

  /* Значения для нескольких фонов */
  /* Не путайте такую запись с background-size: auto auto */
  background-size: auto, auto;
  background-size: 50%, 25%, 25%;
  background-size: 6px, auto, contain;
}
```

<!-- block-size ---------------------------------------------------------------------------------------------------------------------------->

# block-size

Свойство позволяет записать height и width в одно свойство с учетом режима письма writing-mode.

```scss
.block-size {
  block-size: 300px;
  block-size: 25em;

  /* <percentage> values */
  block-size: 75%;

  /* Keyword values */
  block-size: 25em border-box;
  block-size: 75% content-box;
  block-size: max-content;
  block-size: min-content;
  block-size: available;
  block-size: fit-content;
  block-size: auto;
}
```

<!--border  ------------------------------------------------------------------------------------------------------------------------------------>

# border

определит стиль для всех четырех границ сокращенная запись border-width + border-style + border-color

приставки block и inline Добавляют возможность контролировать направление текста

```scss
 {
  border: 4mm ridge rgba(211, 220, 50, 0.6);
}
```

<!--border-bottom-left-right-top  ---------------------------------------------------------------------------------------------------------->

# border-bottom, border-left, border-right, border-top

сокращенная запись для определения стиля, ширины и стиля границы

```scss
 {
  border-bottom: 4mm ridge rgba(211, 220, 50, 0.6);
}
```

# border-collapse (таблицы)

<!--border-collapse------------------------------------------------------------------------------------------------------------------------->

Как ведет себя рамка,по умолчанию есть расстояние между ячейками

```scss
 {
  border-collapse: collapse; //соединить границы
  border-collapse: separate; //разъединить границы таблицы
}
```

<!--border-color  -------------------------------------------------------------------------------------------------------------------------->

# border-left-color-border-right-color-border-top-color

```scss
 {
  border-left-color: red;
  border-left-color: #ffbb00;
  border-left-color: rgb(255 0 0);
  border-left-color: hsl(100deg 50% 25% / 75%);
  border-left-color: currentcolor;
  border-left-color: transparent;
  //короткая запись
  border-color: red yellow green transparent;
}
```

<!-- border-image  ------------------------------------------------------------------------------------------------------------------------------------>

# border-image

Короткая запись для border свойств

border-image-outset + border-image-repeat + border-image-slice + border-image-source + border-image-width

```scss
 {
  border-image: repeating-linear-gradient(30deg, #4d9f0c, #9198e5, #4d9f0c 20px)
    60; //
  border-image: url("/images/border.png") 27 23 / 50px 30px / 1rem round space;
}
```

<!--border-image-outset  -------------------------------------------------------------------------------------------------------------------------->

# border-image-outset

```scss
{
  // от всех границ
  border-image-outset: 1red
  // top | right | bottom | left
  border-image-outset: 7px 12px 14px 5px;
}
```

<!-- border-image-repeat --------------------------------------------------------------------------------------------------------------------->

# border-image-repeat

Позволяет растянуть картинку границы

```scss
 {
  border-image-repeat: stretch; //растяжение изображения
  border-image-repeat: repeat; //повтор
  border-image-repeat: round; //повтор
  border-image-repeat: space; //повтор
  // для нескольких границ
  border-image-repeat: round stretch;
}
```

<!-- border-image-slice ---------------------------------------------------------------------------------------------------------------------->

# border-image-slice

позволяет нарезать на количество кусков картинку и заполнить рамки

```scss
 {
  border-image-slice: 30; //позволяет распределить изображение
  border-image-slice: 30 fill; //fill - заполнит внутреннюю область
}
```

<!-- border-image-source  ------------------------------------------------------------------------------------------------------------------>

# border-image-source

источник изображения

```scss
 {
  border-image-source: url("/media/examples/border-stars.png"); //внутренние ресурсы
  border-image-source: repeating-linear-gradient(
    45deg,
    transparent,
    #4d9f0c 20px
  ); //градиент
  border-image-source: none;
}
```

<!-- border-image-width  ------------------------------------------------------------------------------------------------------------------->

# border-image-width

```scss
 {
  border-image-width: 30px; // в пикселях
  border-image-width: 15px 40px; //для нескольких границ
  border-image-width: 20% 8%; //в процентном соотношении
}
```

<!--border-radius  ------------------------------------------------------------------------------------------------------------------------->

# border-radius-border-bottom-left-radius-border-bottom-right-radius-border-top-left-radius-border-top-right-radius-border-top-right-radius

```scss
 {
  border-bottom-right-radius: 3px;

  border-bottom-right-radius: 20%; //закругление на 1/5 часть края
  border-bottom-right-radius: 20% 10%; //20% от горизонтали и 10% от вертикали
  border-bottom-right-radius: 0.5em 1em;

  // сокращенная запись
  border-radius: 10px;
  /* top-left-and-bottom-right | top-right-and-bottom-left */
  border-radius: 10px 5%;
  /* top-left | top-right-and-bottom-left | bottom-right */
  border-radius: 2px 4px 2px;
  /* top-left | top-right | bottom-right | bottom-left */
  border-radius: 1px 0 3px 4px;
  /* The syntax of the second radius allows one to four values */
  /* (first radius values) / radius */
  border-radius: 10px / 20px;
  /* (first radius values) / top-left-and-bottom-right | top-right-and-bottom-left */
  border-radius: 10px 5% / 20px 30px;
  /* (first radius values) / top-left | top-right-and-bottom-left | bottom-right */
  border-radius: 10px 5px 2em / 20px 25px 30%;
  /* (first radius values) / top-left | top-right | bottom-right | bottom-left */
  border-radius: 10px 5% / 20px 25em 30px 35em;
}
```

<!--  border-spacing ------------------------------------------------------------------------------------------------------------------------>

# border-spacing

Расстояние между ячейками

```scss
.border-spacing {
  /* <length> */
  border-spacing: 2px;

  /* horizontal <length> | vertical <length> */
  border-spacing: 1cm 2em;
}
```

<!--border-style  -------------------------------------------------------------------------------------------------------------------------->

# border-style, border-bottom-style, border-left-style, border-right-style, border-top-style

```scss
 {
  border-bottom-style: none;
  border-bottom-style: hidden; // скрыть
  border-bottom-style: dotted; // в точку
  border-bottom-style: dashed; // в черточку
  border-bottom-style: solid; // сплошной
  border-bottom-style: double; // двойной
  border-bottom-style: groove; // двойной
  border-bottom-style: ridge; // светлый
  border-bottom-style: inset; // без заливки
  border-bottom-style: outset; // с заливкой
  // коротка запись t+r+b+l
  border-style: dashed groove none dotted;
}
```

<!--border-width  -------------------------------------------------------------------------------------------------------------------------->

# border-width, border-bottom-width, border-left-width, border-right-width, border-top-width

```scss
 {
  // текстовые обозначения
  border-bottom-width: thin;
  border-bottom-width: medium;
  border-bottom-width: thick;

  // в абсолютных значения
  border-bottom-width: 10em;
  border-bottom-width: 3vmax;
  border-bottom-width: 6px;
  //сокращенная запись

  border-width: 0 4px 8px 12px;
}
```

<!-- box-decoration-break ------------------------------------------------------------------------------------------------------------------>

# box-decoration-break

```scss
 {
  // при переносе рамка будет разрываться на все строки
  -webkit-box-decoration-break: slice;
  box-decoration-break: slice;
  // при переносе рамка будет оборачивать контент каждой строки
  -webkit-box-decoration-break: clone;
  box-decoration-break: clone;
}
```

<!-- box-shadow ------------------------------------------------------------------------------------------------------------------------------>

# box-shadow

Добавит тень

Параметры:

- значение по горизонтали
- смещение по вертикали
- размытие тени
- цвет тени

```scss
.single-shadow {
  box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.7);
}
```

Множественное значение для теней

```scss
.multiple-shadow {
  box-shadow: 1px 1px 1px black, 2px 2px 1px black, 3px 3px 1px red, 4px 4px 1px
      red, 5px 5px 1px black, 6px 6px 1px black;
}
```

## Значение inset

inset - добавляет внутреннюю тень

```scss
button:active {
  box-shadow: inset 2px 2px 1px black, inset 2px 3px 5px rgba(0, 0, 0, 0.3),
    inset -2px -3px 5px rgba(255, 255, 255, 0.5);
}
```

<!-- box-sizing  --------------------------------------------------------------------------------------------------------------------------->

# box-sizing

определяет как вычисляется величина контейнера.

- если задать ширину и высоту элементу, она будет применена для контента без учета рамок и отступа от рамок

```scss
 {
  //размеры буз учета рамок, стандартное поведение при отступах и рамках реальная ширина будет больше
  box-sizing: content-box;
  width: 100%;
  //будет учитывать размеры отступов
  box-sizing: content-box;
  width: 100%;
  //ужмется по контейнеру
  box-sizing: border-box;
}
```

```scss
div {
  width: 160px;
  height: 80px;
  padding: 20px;
  border: 8px solid red;
  background: yellow;
}

.content-box {
  box-sizing: content-box;
  /* Total width: 160px + (2 * 20px) + (2 * 8px) = 216px
     Total height: 80px + (2 * 20px) + (2 * 8px) = 136px
     Content box width: 160px
     Content box height: 80px */
}

.border-box {
  box-sizing: border-box;
  /* Total width: 160px
     Total height: 80px
     Content box width: 160px - (2 * 20px) - (2 * 8px) = 104px
     Content box height: 80px - (2 * 20px) - (2 * 8px) = 24px */
}
```

<!-- break-after (break-before, break-inside)  --------------------------------------------------------------------------------------------------------------------------->

# break-after (break-before, break-inside)

Применяется для определения разрыва страницы при печати а также для сетки из колонок

break-inside - управление разрывами внутри колонок
break-before, break-inside - до и после

```scss
 {
  break-after: auto; //не будет форсировать разрыв
  break-after: avoid; //избегать любых переносов до/после блока с
  break-after: always;
  break-after: all;

  /* Page break values */
  break-after: avoid-page;
  break-after: page;
  break-after: left;
  break-after: right;
  break-after: recto;
  break-after: verso;

  /* Column break values */
  break-after: avoid-column;
  break-after: column;

  /* Region break values */
  break-after: avoid-region;
  break-after: region;
}
```

# caption-side

<!-- caption-side------------------------------------------------------------------------------------------------------------------------->

```scss
.caption-side: top;
caption-side: bottom;
 {
  caption-side: bottom; // <caption /> будет расположен внизу

  caption-side: top;
  caption-side: bottom;
}
```

<!-- caret-color
 ------------------------------------------------------------------------------------------------------------------------------------------->

# caret-color

```scss
 {
  caret-color: red; //определенный цвет
  caret-color: auto; //обычно current-color
  caret-color: transparent; //невидимая
}
```

<!-- clip-path ------------------------------------------------------------------------------------------------------------------->

# clip-path

```scss
.clip-path {
  clip-path: none;

  /* Значения <clip-source> */
  clip-path: url(resources.svg#c1);

  /* Значения <geometry-box> */
  clip-path: margin-box;
  clip-path: border-box;
  clip-path: padding-box;
  clip-path: content-box;
  clip-path: fill-box;
  clip-path: stroke-box;
  clip-path: view-box;

  /* Значения <basic-shape> */
  clip-path: inset(100px 50px); //Определяет внутренний прямоугольник.
  clip-path: circle(
    50px at 0 100px
  ); //Определяет окружность, используя радиус и расположение.
  clip-path: ellipse(
    50px 60px at 0 10% 20%
  ); //Определяет эллипс, используя два радиуса и расположение
  clip-path: polygon(
    50% 0%,
    100% 50%,
    50% 100%,
    0% 50%
  ); // Определяет многоугольник, используя стиль заполнения фигуры и набор вершин.
  clip-path: path(
    "M0.5,1 C0.5,1,0,0.7,0,0.3 A0.25,0.25,1,1,1,0.5,0.3 A0.25,0.25,1,1,1,1,0.3 C1,0.7,0.5,1,0.5,1 Z"
  ); //Определяет фигуру, используя объявление SVG фигуры и правило заполнения

  /* Комбинация значений границ и формы блока */
  clip-path: padding-box circle(50px at 0 100px);
}
```

```scss
.polygon {
  clip-path: polygon(
    0 0,
    100% 0,
    100% 50%,
    0 100%
  ); //- обрезает картинку по координатам, относительно изображение
}
```

<!-- color ------------------------------------------------------------------------------------------------------------------------------->

# color

```scss
 {
  color: red; //цвет текста
}
```

<!-- color-scheme ---------------------------------------------------------------------------------------------------------------------------->

# color-scheme

Применит стили темы пользователя для элементов

```scss
.color-scheme {
  color-scheme: normal;
  color-scheme: light; //Означает, что элемент может быть отображён в светлой цветовой схеме операционной системы.
  color-scheme: dark; //Означает, что элемент может быть отображён в тёмной цветовой схеме операционной системы.
  color-scheme: light dark;
}

:root {
  color-scheme: light dark;
}
```

<!-- column --------------------------------------------------------------------------------------------------------------------->

# column

Свойство позволяет разделить на столбцы текст в контейнере

```scss
 {
  // разделение текста на две колонки (*)
  -moz-column-count: 2;
  column-count: 2;
  // размер промежутка между колонками (*)
  column-gap: 4rem;
  -moz-column-gap: 4rem;
  // разделитель (*)
  column-rule: 1px solid $color-grey-light-2;
  -moz-column-rule: 1px solid $color-grey-light-2;
  //позволяет растянуть элемент по ширине всех колонок
  column-span: all;
}
```

<!-- column-fill --------------------------------------------------------------------------------------------------------------------------->

# column-fill

```scss
 {
  column-fill: auto; //Высота столбцов не контролируется.
  column-fill: balance; //Разделяет содержимое на равные по высоте столбцы.
}
```

<!-- column-gap (flex, grid, multi-column)---------------------------------------------------------------------------------------------------------------->

# column-gap (flex, grid, multi-column)

расстояние по вертикали

```scss
.column-gap {
  column-gap: auto; //1em
  column-gap: 20px;
}
```

<!-- column-rule  ------------------------------------------------------------------------------------------------------------------------->

# column-rule (multi-column)

Устанавливает цвет границы между колонками = column-rule-width + column-rule-style + column-rule-color

```scss
.column-count {
  // column-count: 3;
  column-rule: solid 8px;
  column-rule: solid blue;
  column-rule: thick inset blue;
}
```

<!-- column-rule-color --------------------------------------------------------------------------------------------------------------------->

# column-rule-color

цвет колонок

```scss
.column-rule-color {
  column-rule-color: red;
  column-rule-color: rgb(192, 56, 78);
  column-rule-color: transparent;
  column-rule-color: hsla(0, 100%, 50%, 0.6);
}
```

<!-- column-rule-style ---------------------------------------------------------------------------------------------------------------------------->

# column-rule-style

Стиль разделителя

```scss
 {
  column-rule-style: none;
  column-rule-style: hidden;
  column-rule-style: dotted;
  column-rule-style: dashed;
  column-rule-style: solid;
  column-rule-style: double;
  column-rule-style: groove;
  column-rule-style: ridge;
  column-rule-style: inset;
  column-rule-style: outset;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# column-rule-width:

Ширина колонки

```scss
 {
  column-rule-width: thin;
  column-rule-width: medium;
  column-rule-width: thick;

  /* <length> values */
  column-rule-width: 1px;
  column-rule-width: 2.5em;
}
```

<!-- column-span ---------------------------------------------------------------------------------------------------------------------------->

# column-span (multi-column)

```scss
.column-span {
  column-span: none;
  column-span: all;
}
```

```html
<!-- контейнер для определения колонок -->
<article>
  <!-- контент для распределения на колонки -->
  <h2>Header spanning all of the columns</h2>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
</article>
```

```scss
article {
  columns: 3;
}

h2 {
  column-span: all;
}
```

<!-- column-width --------------------------------------------------------------------------------------------------------------------->

# column-width (multi-column)

Позволяет определить максимальную ширину колонки

```scss
.container {
  column-width: 200px;
}
```

<!-- columns ---------------------------------------------------------------------------------------------------------------------------->

# columns

Устанавливает количество колонок и их ширину

```scss
 {
  /* количество */
  columns: auto;
  columns: 2;

  /* Количество и ширина */
  columns: 2 auto;
  columns: auto 12em;
  columns: auto auto;
}
```

<!-- contain  ------------------------------------------------------------------------------------------------------------------------------->

# contain

Существует четыре типа ограничения CSS: размер, макет, стиль и краска, которые устанавливаются в контейнере

```scss
 {
  contain: none;
  contain: strict; // === contain: size layout paint style
  contain: content; // === contain: layout paint style блок независимый, невидимые не будет отрисовать
  contain: size; // размер элемента может быть вычислен изолировано, работает в паре с contain-intrinsic-size
  contain: inline-size; // строчное
  contain: layout; // Внутренняя компоновка элемента изолирована от остальной части страницы
  contain: style; //Для свойств, которые могут влиять не только на элемент и его потомков, эффекты не выходят за пределы содержащего элемента
  contain: paint; //Потомки элемента не отображаются за его пределами.
}
```

- В некоторых случаях, особенно при использовании строгого значения strict, браузер может потребовать дополнительных ресурсов для оптимизации рендеринга. Поэтому важно тестировать и измерять производительность при использовании свойства.contain применяется к самому элементу и его содержимому, но не влияет на элементы, вложенные внутри него. Если требуется оптимизировать взаимодействие внутри вложенных элементов, нужно применить свойство contain к каждому из них отдельно.
- Свойство наиболее полезно в ситуациях, когда у вас есть небольшой набор элементов, которые могут быть легко изолированы и оптимизированы.
- В случае сложных макетов с большим количеством элементов, использовать contain бывает сложно и неэффективно

<!-- contain-intrinsic- ---------------------------------------------------------------------------------------------------------------------------->

# contain-intrinsic-block-size | block-height | inline-size | intrinsic-size | intrinsic-width

Настройка размеров блочных и строчных элементов при ограничении

contain-intrinsic-size = contain-intrinsic-width + contain-intrinsic-height

```scss
.contain-intrinsic {
  contain-intrinsic-block-size: 1000px;
  contain-intrinsic-block-size: 10rem;
  contain-intrinsic-height: 1000px;
  contain-intrinsic-height: 10rem;
  contain-intrinsic-inline-size: 1000px;
  contain-intrinsic-inline-size: 10rem;

  /* auto <length> */
  contain-intrinsic-block-size: auto 300px;
  contain-intrinsic-height: auto 300px;
  contain-intrinsic-inline-size: auto 300px;
}
```

<!-- container ----------------------------------------------------------------------------------------------------------------------------->

# container

container = container-name + container-type

```scss
.container {
  container: my-layout;
  container: my-layout / size;
}
```

<!-- container-name ------------------------------------------------------------------------------------------------------------------------->

# container-name

Определяет имя контейнера

```scss
 {
  container-name: myLayout;
  container-name: myPageLayout myComponentLibrary; //несколько имен
}
```

<!--  ------------------------------------------------------------------------------------------------------------------------------------>

# container-type

```scss
 {
  container-type: normal;
  container-type: size; //по inline и block модели
  container-type: inline-size; //по строчной
}
```

<!-- content ------------------------------------------------------------------------------------------------------------------------------->

# content

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

  /* Значения <quote> */
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

<!-- content-visibility -------------------------------------------------------------------------------------------------------------------->

# content-visibility

Позволяет сделать содержимое контейнера невидимым. Основное применение для создание плавных анимаций, при которых контент плавно пропадает.
В анимации нужно включить transition-behavior: content-visibility

```scss
 {
  content-visibility: visible; //обычное отображение элемента
  content-visibility: hidden; // не будет доступно для поиска, фокусировки
  content-visibility: auto; //contain: content
}
```

Второе применение экономия ресурсов при рендеринге

<!--  ------------------------------------------------------------------------------------------------------------------------------------>

# counter-increment, counter-set, counter-reset,

используется для увеличения значений в списке

```scss
// сброс счетчика
div {
  counter-reset: my-counter 100; //задает новое значение
}
div {
  // объявляем счетчик и начальное значение по умолчанию ноль
  counter-increment: my-counter -1;
}
div {
  // объявляем счетчик и начальное значение по умолчанию ноль
  counter-set: my-counter -1; //задает новое значение
}
i::before {
  // запуск c помощью функции counter
  content: counter(sevens);
}
```

-[функция counter()](./functions.md)

список, который уменьшается на 1

```html
<div>
  <i>1</i>
  <i>100</i>
</div>
```

<!-- cursor ------------------------------------------------------------------------------------------------------------------------------>

# cursor

Определяет тип курсора

```scss
 {
  // определенные системой их много полный список https://developer.mozilla.org/en-US/docs/Web/CSS/cursor
  cursor: auto;
  cursor: pointer;
  cursor: help;
  cursor: wait;
  cursor: crosshair;
  cursor: not-allowed;
  cursor: zoom-in;
  cursor: grab;
  // пользовательский
  cursor: url("hyper.cur"), auto;
  // определение положения
  cursor: url("hyper.cur") 0 0;
}
```

<!-- display-------------------------------------------------------------------------------------------------------------------------------->

# display

```scss
.display {
  /* <display-outside> values */
  display: block;
  display: inline;
  display: run-in;

  /* <display-inside> values */
  display: flow;
  display: flow-root;
  display: table;
  display: flex;
  display: grid;
  display: ruby;

  /* <display-outside> plus <display-inside> values */
  display: block flow;
  display: inline table;
  display: flex run-in;

  // списковые
  display: list-item;
  display: list-item block;
  display: list-item inline;
  display: list-item flow;
  display: list-item flow-root;
  display: list-item block flow;
  display: list-item block flow-root;
  display: flow list-item block;

  // табличные
  display: table-row-group;
  display: table-header-group;
  display: table-footer-group;
  display: table-row;
  display: table-cell;
  display: table-column-group;
  display: table-column;
  display: table-caption;
  display: ruby-base;
  display: ruby-text;
  display: ruby-base-container;
  display: ruby-text-container;

  /* <display-box> values */
  display: contents; //создаст псевдо-контейнер по своим дочерним элементам
  display: none; //удаляем из дерева

  /* <display-legacy> values */
  display: inline-block;
  display: inline-table;
  display: inline-flex;
  display: inline-grid;

  /* Global values */
  display: inherit;
  display: initial;
  display: unset;
}
```

<!-- empty-cells --------------------------------------------------------------------------------------------------------------------------->

# empty-cells

Показывать или нет пустые ячейки

```scss
.empty-cells {
  empty-cells: show | hide;
}
```

<!-- filter ------------------------------------------------------------------------------------------------------------------->

# filter

Добавляет фильтры на изображения

```scss
{
  filter: url(resources.svg);
  filter: blur(5px);
  filter: brightness(0.4);
  filter: contrast(200%);
  filter: drop-shadow(16px 16px 20px blue);
  filter: grayscale(50%);
  filter: hue-rotate(90deg);
  filter: invert(75%);
  filter: opacity(25%);
  filter: saturate(30%);
  filter: sepia(60%);
  // fill – заливка
  fill: currentColor; заливка цветом
}
```

```scss
img {
}

.blur {
  filter: blur(10px);
}
```

```html
<div class="box"><img src="balloons.jpg" alt="balloons" class="blur" /></div>
```

Фильтр можно добавить к объектам, то есть к самой тени

```scss
p {
  border: 5px dashed red;
}

.filter {
  filter: drop-shadow(5px 5px 1px rgb(0 0 0 / 70%));
}

.box-shadow {
  box-shadow: 5px 5px 1px rgb(0 0 0 / 70%);
}
```

<!-- flex -------------------------------------------------------------------------------------------------------------------------------->

# flex

flex-grow: 0
flex-shrink: 1
flex-basis: auto

```scss
// шорткат для flex-grow + flex-shrink + flex-basis
 {
  flex: 1 1 200px;
}
```

<!-- flex-basis ---------------------------------------------------------------------------------------------------------------------------->

# flex-basis

Устанавливает минимальное значение размера flex- элемента, если оно не установлено блочной моделью.

Разница между flex-basis 0 и 0% в том что, во втором случает элемент ужмется до своих минимальных размеров внутреннего контента

```scss
 {
  flex-basis: auto; // значение по умолчанию
  flex-basis: fill;
  flex-basis: max-content;
  flex-basis: min-content;
  flex-basis: fit-content;
  flex-basis: content; // определяет размер на основе содержимого
  flex-basis: 0; // определяет пропорционально с другими элементами
  flex-basis: 100px; // если в px то определяет минимальный размер контейнера
}
```

# flex-direction

```scss
 {
  flex-direction: row; // справа на лево, то есть блоки будут идти справа на лево
  flex-direction: column; // сверху вниз, как div-ы
  flex-direction: row-reverse; // снизу вверх
  flex-direction: column-reverse;
  // общие значения
  flex-direction: inherit;
  flex-direction: initial;
  flex-direction: revert;
  flex-direction: revert-layer;
  flex-direction: unset;
}
```

<!-- flex-flow ----------------------------------------------------------------------------------------------------------------------------->

# flex-flow

```scss
 {
  // --------------------------------------------------------------------
  // позволяет задать в одной строчке задать flex-direction + flex-wrap
  flex-flow: row wrap;
}
```

<!-- flex-wrap ----------------------------------------------------------------------------------------------------------------------------->

# flex-wrap

```scss
 {
  flex-wrap: wrap; //
}
```

<!-- font -------------------------------------------------------------------------------------------------------------------------------->

# font

Сокращенная запись для я font-style, font-variant, font-weight, font-stretch, font-size, line-height, и font-family

```scss
{
  font-family: "Arial", "Helvetica"; //для форматирования шрифта, если на компьютере не установлен "Arial", то система попытается найти "Helvetica", serif и sans-serif – универсальные
// https://fonts.google.com/ - для поиска шрифтов
font-size: 36px; //- размер текста в пикселах, стандартный размер 16px
font-size: 1em; //- размер относительно 16px (размер заглавной буквы M)
font-size: 1rem; //- размер относительно базового элемента
font-size: 100%; //- размер относительно 16px только в процентах
font-size: "xx-small (9px)" | "x-small" | "xx-large (32px)"; //- размер
// --------------------------------------------------------------------
font-weight: "normal" | "bold" | "lighter" | "bolder" | 100-900 //– жирность шрифта
font-style: "normal" | "oblique" | "italic" //– наклонение шрифта
font-variant: //– типографические эффекты
}

```

<!-- font-family---------------------------------------------------------------------------------------------------------------------------->

# font-family

Определяет приоритетность шрифта

```scss
 {
  // оба определения валидные
  font-family: Gill Sans Extrabold, sans-serif;
  font-family: "Goudy Bookletter 1911", sans-serif;

  /* Только общие семейства */
  font-family: serif; //со штрихами
  font-family: sans-serif; //гладкие
  font-family: monospace; //одинаковая ширина
  font-family: cursive; //рукопись
  font-family: fantasy; //декор-ые
  font-family: system-ui; //из системы
  font-family: emoji; //
  font-family: math; //
  font-family: fangsong; //китайский
}
```

<!-- font-size  ---------------------------------------------------------------------------------------------------------------------------->

# font-size

Свойство для изменения размера

```scss
 {
  /* значения в <абсолютных размерах> */
  font-size: xx-small;
  font-size: x-small;
  font-size: small;
  font-size: medium;
  font-size: large;
  font-size: x-large;
  font-size: xx-large;
  /* значения в <относительных размерах> */
  font-size: larger;
  font-size: smaller;
  font-size: 12px;
  font-size: 0.8em;
  font-size: 80%;
}
```

Масштабирование с помощью font-size

```scss
body {
  font-size: 62.5%; /* font-size 1em = 10px on default browser settings */
}

span {
  font-size: 1.6em; /* 1.6em = 16px */
}
```

<!-- font-style ---------------------------------------------------------------------------------------------------------------------------->

# font-style

Стили шрифтов

```scss
 {
  font-style: normal;
  font-style: italic; //курсив
  font-style: oblique; //курсив
}
```

<!--font-weight  --------------------------------------------------------------------------------------------------------------------------->

# font-weight

Начертание шрифта

```scss
 {
  /font-weight: normal;
  font-weight: bold;

  /* Relative to the parent */
  font-weight: lighter;
  font-weight: bolder;

  font-weight: 100;
  font-weight: 200;
  font-weight: 300;
  font-weight: 400;
  font-weight: 500;
  font-weight: 600;
  font-weight: 700;
  font-weight: 800;
  font-weight: 900;
}
```

<!-- gap (flex, grid)------------------------------------------------------------------------------------------------------------------->

# gap (flex, grid)

сокращенная запись gap = row-gap + column-gap

```scss
 {
  gap: 10px 20px;
}
```

<!-- grid ---------------------------------------------------------------------------------------------------------------------------->

# grid

Является сокращением для следующих свойств (значения по умолчанию)

```scss
.grid {
  grid-template-rows: none;
  grid-template-columns: none;
  grid-template-areas: none;
  grid-auto-rows: auto;
  grid-auto-columns: auto;
  grid-auto-flow: row;
  grid-column-gap: 0;
  grid-row-gap: 0;
  column-gap: normal;
  row-gap: normal;

  // варианты

  grid: none;
  grid: "a" 100px "b" 1fr;
  grid: [linename1] "a" 100px [linename2];
  grid: "a" 200px "b" min-content;
  grid: "a" minmax(100px, max-content) "b" 20%;
  grid: 100px / 200px;
  grid: minmax(400px, min-content) / repeat(auto-fill, 50px);

  /* <'grid-template-rows'> /
   [ auto-flow && dense? ] <'grid-auto-columns'>? values */
  grid: 200px / auto-flow;
  grid: 30% / auto-flow dense;
  grid: repeat(3, [line1 line2 line3] 200px) / auto-flow 300px;
  grid: [line1] minmax(20em, max-content) / auto-flow dense 40%;

  /* [ auto-flow && dense? ] <'grid-auto-rows'>? /
   <'grid-template-columns'> values */
  grid: auto-flow / 200px;
  grid: auto-flow dense / 30%;
  grid: auto-flow 300px / repeat(3, [line1 line2 line3] 200px);
  grid: auto-flow dense 40% / [line1] minmax(20em, max-content);
}
```

<!-- grid-auto-rows ------------------------------------------------------------------------------------------------------------------------>

# grid-auto-rows и grid-auto-columns

grid-auto-rows - для автоматического распределения ширины
grid-auto-columns - длины элемента

```scss
 {
  // автоматическое распределение
  grid-auto-rows: min-content;
  grid-auto-rows: max-content;
  grid-auto-rows: auto;
  //поддерживает проценты, пиксели, функции min-max
  // для сетки с множеством колонок или строк (если перенесется более одной строки)
  rid-auto-rows: min-content max-content auto;
}
```

Для автоматического определения высоты в строках неявной сетки

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  grid-auto-rows: minmax(100px, auto);
}
```

<!-- grid-auto-flow ------------------------------------------------------------------------------------------------------------------------------------>

# grid-auto-flow

Определяет размещение элементов в неявной grid сетке

```scss
 {
  grid-auto-flow: row; //вынесет элементы в новый ряд
  grid-auto-flow: column; //вынесет элементы в новую колонку
  grid-auto-flow: dense; //автоматическое распределение
  grid-auto-flow: row dense;
  grid-auto-flow: column dense;
}
```

<!-- grid-column-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-column-gap

```scss
.grid-column-gap {
  grid-column-gap: 10px;
}
```

<!-- grid-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-gap

```scss
.grid-gap {
  rid-gap: 10px 12px;
}
```

<!-- grid-row-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-row-gap

```scss
.grid-row-gap {
  grid-row-gap: 10px;
}
```

<!--grid-template = grid-template-columns + grid-template-rows ------------------------------------------------------------------------------------------------------------------------------------>

# grid-template = grid-template-columns + grid-template-rows

Позволяет сформировать макет с помощью контейнера

```scss
 {
  grid-template-columns: 100px 1fr;
  grid-template-columns: [linename] 100px;
  grid-template-columns: [linename1] 100px [linename2 linename3];
  grid-template-columns: minmax(100px, 1fr);
  grid-template-columns: fit-content(40%);
  grid-template-columns: repeat(3, 200px);
  grid-template-columns: subgrid;
  grid-template-columns: masonry;

  /* <auto-track-list> values */
  grid-template-columns: 200px repeat(auto-fill, 100px) 300px;
  grid-template-columns:
    minmax(100px, max-content)
    repeat(auto-fill, 200px) 20%;
  grid-template-columns:
    [linename1] 100px [linename2]
    repeat(auto-fit, [linename3 linename4] 300px)
    100px;
  grid-template-columns:
    [linename1 linename2] 100px
    repeat(auto-fit, [linename1] 300px) [linename3];
}
```

<!-- height -------------------------------------------------------------------------------------------------------------------------------->

# height

```scss
 {
  // если в процентах, то от контейнера
  height: 120px;
  height: 10em;
  height: 100vh;
  height: anchor-size(height);
  height: anchor-size(--myAnchor self-block, 250px);
  height: clamp(200px, anchor-size(width));

  /* <percentage> value */
  height: 75%;

  /* Keyword values */
  height: max-content;
  height: min-content;
  height: fit-content;
  height: fit-content(20em);
  height: auto;
  height: minmax(min-content, anchor-size(width));
  height: stretch;
}
```

<!-- hyphens --------------------------------------------------------------------------------------------------------------------->

# hyphens

указывает, как следует переносить слова через дефис, когда текст переносится на несколько строк

```scss
 {
  hyphens: none;
  hyphens: manual;
  hyphens: auto;
  -moz-hyphens: auto;
  -ms-hyphens: auto;
  -webkit-hyphens: auto;
  //правильный разделитель слов (*)
  hyphens: auto;
}
```

<!-- hyphenate-character ------------------------------------------------------------------------------------------------------------------->

# hyphenate-character

```scss
.hyphenate-character {
  hyphenate-character: <string>;
  hyphenate-character: auto;
}
```

```html
<dl>
  <dt><code>hyphenate-character: "="</code></dt>
  <dd id="string" lang="en">Superc&shy;alifragilisticexpialidocious</dd>
  <dt><code>hyphenate-character is not set</code></dt>
  <dd lang="en">Superc&shy;alifragilisticexpialidocious</dd>
</dl>
```

```scss
dd {
  width: 90px;
  border: 1px solid black;
  hyphens: auto;
}

dd#string {
  -webkit-hyphenate-character: "=";
  hyphenate-character: "=";
}
```

<!-- inset-area -------------------------------------------------------------------------------------------------------------------------->

# inset-area (anchor)

Нестабильное свойство. Позволяет позиционировать якорь

![расположение якоря](../css-assets/inset-area.png)

```scss
 {
  inset-area: none;

  //  для одного заголовка
  inset-area: top left;
  inset-area: start end;
  inset-area: block-start center;
  inset-area: inline-start block-end;
  inset-area: x-start y-end;
  inset-area: center y-self-end;
  // для двух
  inset-area: top span-left;
  inset-area: center span-start;
  inset-area: inline-start span-block-end;
  inset-area: y-start span-x-end;
  // для трех
  inset-area: top span-all;
  inset-area: block-end span-all;
  inset-area: x-self-start span-all;

  /* One <inset-area> keyword with an implicit second <inset-area> keyword  */
  inset-area: top; /* equiv: top span-all */
  inset-area: inline-start; /* equiv: inline-start span-all */
  inset-area: center; /* equiv: center center */
  inset-area: span-all; /* equiv: center center */
  inset-area: end; /* equiv: end end */

  /* Global values */
  inset-area: inherit;
  inset-area: initial;
  inset-area: revert;
  inset-area: revert-layer;
  inset-area: unset;
}
```

<!-- inset- ------------------------------------------------------------------------------------------------------------------------------------>

# inset- (якоря)

```scss
 {
  inset-block-start: 3px | 1rem | anchor(end) | calc(
      anchor(--myAnchor 50%) + 5px
    )
    | 10%
    // расположит элемент якоря, аналогично inset-block-end ,inset-inline-start, inset-inline-end
;
  inset-block: 10px 20px; // определяет начальные и конечные смещения логического блока элемента, аналогично inset-inline
  inset: ; // inset-block-start + inset-block-end + inset-inline-start + inset-inline-end
}
```

<!-- isolation ----------------------------------------------------------------------------------------------------------------------------->

# isolation

Управление контекстом стекирования

```scss
 {
  isolation: auto;
  isolation: isolate;
}
```

<!-- justify-content (flex) --------------------------------------------------------------------------------------------------------------------->

# justify-content (flex, grid)

```scss
.flex {
  justify-content: center; // Выравнивание элементов по центру
  justify-content: start; // Выравнивание элементов в начале в отличие от flex-start отчет идет от направления письма
  justify-content: end; // Выравнивание элементов в конце
  justify-content: flex-start; // Выравнивание флекс-элементов с начала
  justify-content: flex-end; // Выравнивание флекс-элементов с конца
  justify-content: left; // Выравнивание элементов по левому краю
  justify-content: right; // Выравнивание элементов по правому краю

  // Выравнивание относительно осевой линии
  justify-content: baseline;
  justify-content: first baseline;
  justify-content: last baseline;

  // Распределённое выравнивание
  justify-content: space-between; // Равномерно распределяет все элементы по ширине flex-блока. Первый элемент вначале, последний в конце
  justify-content: space-around; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют полноразмерное пространство с обоих концов
  justify-content: space-evenly; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют равное пространство вокруг
  justify-content: stretch; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют "авто-размер", чтобы соответствовать контейнеру
  // Выравнивание при переполнении
  justify-content: safe center;
  justify-content: unsafe center;
}
```

<!-- justify-item (grid)-------------------------------------------------------------------------------------------------------------------------->

# justify-item (grid)

игнорируется в таблицах, flex и grid сетках

```scss
 {
  justify-items: center; // Выровнять элементы по центру
  justify-items: start; // Выровнять элементы в начале
  justify-items: end; // Выровнять элементы в конце
  justify-items: flex-start; // Эквивалентно 'start'. Обратите внимание, что justify-items игнорируется в разметке Flexbox.
  justify-items: flex-end; // Эквивалентно 'end'. Обратите внимание, что justify-items игнорируется в разметке Flexbox.
  justify-items: self-start;
  justify-items: self-end;
  justify-items: left; // Выровнять элементы по левому краю
  justify-items: right; // Выровнять элементы по правому краю
  /* Исходное выравнивание */
  justify-items: baseline;
  justify-items: first baseline;
  justify-items: last baseline;
  /* Выравнивание при переполнении (только для выравнивания положения) */
  justify-items: safe center;
  justify-items: unsafe center;
  /* Унаследованное выравнивание */
  justify-items: legacy right;
  justify-items: legacy left;
  justify-items: legacy center;
}
```

<!-- justify-self (grid) ---------------------------------------------------------------------------------------------------------------------------->

# justify-self (grid)

выравнивание элемент вдоль главной оси. не работает в flex и табличных контейнерах

```scss
 {
  // Positional alignment
  justify-self: center; // Pack item around the center
  justify-self: start; // Pack item from the start
  justify-self: end; // Pack item from the end
  justify-self: flex-start; // Equivalent to 'start'. Note that justify-self is ignored in flexbox layouts.
  justify-self: flex-end; // Equivalent to 'end'. Note that justify-self is ignored in flexbox layouts.
  justify-self: self-start;
  justify-self: self-end;
  justify-self: left; // Pack item from the left
  justify-self: right; // Pack item from the right
  justify-self: anchor-center;

  // Baseline alignment
  justify-self: baseline;
  justify-self: first baseline;
  justify-self: last baseline;

  // Overflow alignment (for positional alignment only)
  justify-self: safe center;
  justify-self: unsafe center;
}
```

<!-- letter-spacing ---------------------------------------------------------------------------------------------------------------------->

# letter-spacing

расстояние между буквами

```scss
 {
  letter-spacing: "px", "%";
}
```

<!-- line-height --------------------------------------------------------------------------------------------------------------------->

# line-height

расстояние между строками

```scss
 {
  line-height: "px", "%";
}
```

<!-- list --------------------------------------------------------------------------------------------------------------------->

# list-style

Сокращенная запись для list-style = list-style-image + list-style-position + list-style-type

```scss
 {
  //тип маркеров
  list-style-type: disc;
  list-style-type: circle;
  list-style-type: square;
  list-style-type: decimal;
  list-style-type: georgian;
  list-style-type: trad-chinese-informal;
  list-style-type: kannada;
  list-style-type: "-";
  /* Identifier matching an @counter-style rule */
  list-style-type: custom-counter-style;
  list-style-type: none;
  list-style-type: inherit;
  list-style-type: initial;
  list-style-type: revert;
  list-style-type: revert-layer;
  list-style-type: unset;

  //где будет располагаться
  list-style-position: inside; //::marker перед контентом
  list-style-position: outside; //::marker внутри контента

  //изображение
  list-style-image: url(example.png);

  // шорткат
  list-style: square url(example.png) inside; // list-style-type list-style-image list-style-position
}
```

<!-- list-style-image  --------------------------------------------------------------------------------------------------------------------->

# list-style-image

Позволяет добавить изображение в список в качестве разделителя

```scss
 {
  list-style-image: none;

  /* <url> значения */
  list-style-image: url("starsolid.gif");
}
```

<!-- margin
  ---------------------------------------------------------------------------------------------------------------------------------->

# margin

приставки block и inline Добавляют возможность контролировать направление текста

```scss
 {
  margin: auto; // Прием позволяет отдать под отступ все доступное пространство
}
```

<!-- mask ---------------------------------------------------------------------------------------------------------------------------------->

# mask

mask = mask-clip + mask-composite + mask-image + mask-mode + mask-origin + mask-position + mask-repeat + mask-size

```scss
 {
  //
}
```

<!-- mask-clip ----------------------------------------------------------------------------------------------------------------------------->

# mask-clip

определяет область применения маски

```scss
 {
  mask-clip: content-box;
  mask-clip: padding-box;
  mask-clip: border-box;
  mask-clip: fill-box;
  mask-clip: stroke-box;
  mask-clip: view-box;

  /* Keyword values */
  mask-clip: no-clip;

  /* Non-standard keyword values */
  -webkit-mask-clip: border;
  -webkit-mask-clip: padding;
  -webkit-mask-clip: content;
  -webkit-mask-clip: text;

  /* Multiple values */
  mask-clip: padding-box, no-clip;
  mask-clip: view-box, fill-box, border-box;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# mask-image

ресурс для маски

```scss
 {
  mask-image: url(masks.svg#mask1);

  /* <image> values */
  mask-image: linear-gradient(rgb(0 0 0 / 100%), transparent);
  mask-image: image(url(mask.png), skyblue);

  /* Multiple values */
  mask-image: image(url(mask.png), skyblue), linear-gradient(rgb(0 0 0 / 100%), transparent);
}
```

<!-- mask-origin --------------------------------------------------------------------------------------------------------------------------->

# mask-origin

```scss
 {
  mask-origin: content-box; // Положение указывается относительно границы поля.
  mask-origin: padding-box; //Положение указывается относительно ограничивающей рамки объекта.
  mask-origin: border-box; //Положение указывается относительно ограничивающей рамки штриха.
  mask-origin: fill-box; //Использует ближайший вьюпорт SVG
  mask-origin: stroke-box; //
  mask-origin: view-box;

  /* Multiple values */
  mask-origin: padding-box, content-box;
  mask-origin: view-box, fill-box, border-box;

  /* Non-standard keyword values */
  -webkit-mask-origin: content; //content-box
  -webkit-mask-origin: padding; //padding-box.
  -webkit-mask-origin: border; //border-box.
}
```

<!-- mask-repeat ---------------------------------------------------------------------------------------------------------------------------->

# mask-repeat

Определение повторение маски

```scss
 {
  mask-repeat: repeat-x;
  mask-repeat: repeat-y;
  mask-repeat: repeat;
  mask-repeat: space;
  mask-repeat: round;
  mask-repeat: no-repeat;

  /* Two-value syntax: horizontal | vertical */
  mask-repeat: repeat space;
  mask-repeat: repeat repeat;
  mask-repeat: round space;
  mask-repeat: no-repeat round;
}
```

<!-- mask-size ---------------------------------------------------------------------------------------------------------------------------->

# mask-size

Размер маски

```scss
 {
  /* Keywords syntax */
  mask-size: cover;
  mask-size: contain;

  /* One-value syntax */
  /* the width of the image (height set to 'auto') */
  mask-size: 50%;
  mask-size: 3em;
  mask-size: 12px;
  mask-size: auto;

  /* Two-value syntax */
  /* first value: width of the image, second value: height */
  mask-size: 50% auto;
  mask-size: 3em 25%;
  mask-size: auto 6px;
  mask-size: auto auto;

  /* Multiple values */
  /* Do not confuse this with mask-size: auto auto */
  mask-size: auto, auto;
  mask-size: 50%, 25%, 25%;
  mask-size: 6px, auto, contain;
}
```

<!-- mix-blend-mode  ---------------------------------------------------------------------------------------------------------------------->

# mix-blend-mode

правило смешивания наслаивающих изображений и фонов https://developer.mozilla.org/en-US/docs/Web/CSS/blend-mode

```scss
 {
  mix-blend-mode: lighten | overlay;
}
```

<!-- object-fit ------------------------------------------------------------------------------------------------------------------------------>

# object-fit

Позволяет тегу img определить размеры относительно контейнера

```scss
.object-fit {
  object-fit: fill; //заполняет весь контейнер, меняя свои пропорции
  object-fit: contain; //растянет под контейнер, но оставит пропорции
  object-fit: cover; //оставит пропорции, но поместит в контейнер часть изображения
  object-fit: none; //подстроится под изображение
  object-fit: scale-down; //выберет меньший между none и contain
}
```

<!-- object-position ----------------------------------------------------------------------------------------------------------------------->

# object-position

расположит изображение в контейнере

```scss
 {
  object-position: center top;
  object-position: 100px 50px;
}
```

<!-- offset ---------------------------------------------------------------------------------------------------------------------------->

# offset

```scss
 {
  offset: 10px 30px;

  /* Offset path */
  offset: ray(45deg closest-side);
  offset: path("M 100 100 L 300 100 L 200 300 z");
  offset: url(arc.svg);

  /* Offset path with distance and/or rotation */
  offset: url(circle.svg) 100px;
  offset: url(circle.svg) 40%;
  offset: url(circle.svg) 30deg;
  offset: url(circle.svg) 50px 20deg;

  /* Including offset anchor */
  offset: ray(45deg closest-side) / 40px 20px;
  offset: url(arc.svg) 2cm / 0.5cm 3cm;
  offset: url(arc.svg) 30deg / 50px 100px;
}
```

<!-- offset-anchor ---------------------------------------------------------------------------------------------------------------------------->

# offset-anchor

Позволяет определить где будет находится элемент относительно прямой при движение по линии [offset](./css-props.md/#offset)

```scss
 {
  offset-anchor: top;
  offset-anchor: bottom;
  offset-anchor: left;
  offset-anchor: right;
  offset-anchor: center;
  offset-anchor: auto;

  /* <percentage> values */
  offset-anchor: 25% 75%;

  /* <length> values */
  offset-anchor: 0 0;
  offset-anchor: 1cm 2cm;
  offset-anchor: 10ch 8em;

  /* Edge offsets values */
  offset-anchor: bottom 10px right 20px;
  offset-anchor: right 3em bottom 10px;
}
```

<!-- offset-path --------------------------------------------------------------------------------------------------------------------------->

# offset-path

Позволяет задать путь движения

```scss
 {
  offset-path: ray(45deg closest-side contain);
  offset-path: ray(contain 150deg at center center);
  offset-path: ray(45deg);

  /* URL */
  offset-path: url(#myCircle);

  /* Basic shape */
  offset-path: circle(50% at 25% 25%);
  offset-path: ellipse(50% 50% at 25% 25%);
  offset-path: inset(50% 50% 50% 50%);
  offset-path: polygon(30% 0%, 70% 0%, 100% 50%, 30% 100%, 0% 70%, 0% 30%);
  offset-path: path(
    "M 0,200 Q 200,200 260,80 Q 290,20 400,0 Q 300,100 400,200"
  );
  offset-path: rect(5px 5px 160px 145px round 20%);
  offset-path: xywh(0 5px 100% 75% round 15% 0);

  /* Coordinate box */
  offset-path: content-box;
  offset-path: padding-box;
  offset-path: border-box;
  offset-path: fill-box;
  offset-path: stroke-box;
  offset-path: view-box;
}
```

<!-- orphans ------------------------------------------------------------------------------------------------------------------------------->

# orphans

Минимальное число строк, которое можно оставить внизу фрагмента перед разрывом фрагмента. Значение должно быть положительным.

```scss
 {
  orphans: 3;
}
```

<!-- outline ----------------------------------------------------------------------------------------------------------------------------->

# outline

Свойство обводки контента outline = outline-color + outline-style + outline-width
outline-offset

```scss
 {
  outline: 8px ridge rgba(170, 50, 220, 0.6);
}
```

<!-- outline-color ------------------------------------------------------------------------------------------------------------------------->

# outline-color

```scss
 {
  outline-color: red;
}
```

<!-- outline-offset ------------------------------------------------------------------------------------------------------------------------>

# outline-offset

отступ от обводки внешней границы

```scss
 {
  outline-offset: 4px;
  outline-offset: 0.6rem;
}
```

<!-- outline-style ------------------------------------------------------------------------------------------------------------------------->

# outline-style

стиль внешней обводки

```scss
 {
  outline-style: auto;
  outline-style: none;
  outline-style: dotted;
  outline-style: dashed;
  outline-style: solid;
  outline-style: double;
  outline-style: groove;
  outline-style: ridge;
  outline-style: inset;
  outline-style: outset;
}
```

<!-- outline-width ------------------------------------------------------------------------------------------------------------------------->

# outline-width

ширина внешней обводки

```scss
 {
  // предопределенные стили
  outline-width: thin;
  outline-width: medium;
  outline-width: thick;
  // пользовательские
  outline-width: 1px;
  outline-width: 0.1em;
}
```

<!-- overflow -------------------------------------------------------------------------------------------------------------------------------->

# overflow

overflow-block, overflow-inline - Для rtl

```scss
.overflow {
  // При превышении размера контента используется свойство overflow
  overflow: visible; //(по умолчанию) – не воспрепятствует налеганию текста друг на друга
  overflow: scroll; //добавляет полосы прокрутки
  overflow: auto; //полосы прокрутки появляются при необходимости
  overflow: hidden; //скрывает любое содержимое выходящее за рамки
  overflow-y: scroll; // скролл по вертикали
  overflow-x: scroll; // скролл по горизонтали
}
```

<!-- overflow-wrap ---------------------------------------------------------------------------------------------------------------------------->

# overflow-wrap

разрыв сплошных строк при переносе

```scss
 {
  overflow-wrap: normal;
  overflow-wrap: break-word; //мягкий разрыв предусматривается
  overflow-wrap: anywhere; //мягкий разрыв не предусматривается
}
```

<!-- padding  ------------------------------------------------------------------------------------------------------------------------------>

# padding

приставки block и inline Добавляют возможность контролировать направление текста

```scss
 {
  //
}
```

<!-- page-break-before --------------------------------------------------------------------------------------------------------------------->

# page-break-before, page-break-after, page-break-inside

Устанавливает разрывы для печати на странице до или после элемента

```scss
 {
  page-break-before: auto;
  page-break-before: always;
  page-break-before: avoid;
  page-break-before: left;
  page-break-before: right;
  page-break-before: recto;
  page-break-before: verso;
}
```

<!-- place-items (grid fle-------------------------------------------------------------------------------------------------------------->

# place-items (grid, flex)

короткая запись place-items = align-items + justify-items

```scss
 {
  place-items: end center;
}
```

<!-- perspective-origin -------------------------------------------------------------------------------------------------------------------->

# perspective-origin

```scss
.perspective-origin {
  perspective-origin: x-position;

  /* Two-value syntax */
  perspective-origin: x-position y-position;

  /* When both x-position and y-position are keywords,
   the following is also valid */
  perspective-origin: y-position x-position;
}
```

<!-- place-self (grid, flex) --------------------------------------------------------------------------------------------------------------->

# place-self (grid, flex)

place-self = align-self + justify-self

```scss
 {
  place-self: stretch center;
}
```

<!-- pointer-events ------------------------------------------------------------------------------------------------------------------------>

# pointer-events

Определяет цель для курсора

```scss
 {
  pointer-events: auto;
  pointer-events: none;
  // для svg
  pointer-events: visiblePainted;
  pointer-events: visibleFill;
  pointer-events: visibleStroke;
  pointer-events: visible;
  pointer-events: painted;
  pointer-events: fill;
  pointer-events: stroke;
  pointer-events: bounding-box;
  pointer-events: all;
}
```

<!-- position -------------------------------------------------------------------------------------------------------------------------------->

# position

```scss
 {
  //
  position: static; //нормальное расположение
  position: relative; //позиционирует элементы относительно своей нормальной позиции, с возможностью наехать на другой элемент
  position: absolute; //вытаскивает элемент из нормального потока
  position: fixed; //остается на одном и том же месте
  position: sticky; // ведет себя как static пока не достигнет края окна во время прокрутки
}
```

<!--  position-anchor (якоря) ------------------------------------------------------------------------------------------------------------>

# position-anchor (якоря)

Ограниченная доступность. Определяет имя якоря элемента. Актуально только для позиционированных элементов

```scss
 {
  position-anchor: --anchorName; //имя якоря
}
```

<!-- row-gap (flex, grid)------------------------------------------------------------------------------------------------------------------->

# row-gap (flex, grid)

расстояние по горищонтали

```scss
 {
  row-gap: 20px;
}
```

<!-- resize -------------------------------------------------------------------------------------------------------------------------------->

# resize

Позволяет растягивать элемент

!!! не поддерживается safari

```scss
 {
  resize: none; //отключает растягивание
  resize: both; //тянуть можно во все стороны
  resize: horizontal;
  resize: vertical;
  resize: block; // в зависимости от writing-mode и direction
  resize: inline; // в зависимости от writing-mode и direction
}
```

<!-- rotate -------------------------------------------------------------------------------------------------------------------------------->

# rotate

Позволяет вращать 3-d объект

```scss
.rotate {
  //* Angle value */
  rotate: 90deg;
  rotate: 0.25turn;
  rotate: 1.57rad;

  /* x, y, or z axis name plus angle */
  rotate: x 90deg;
  rotate: y 0.25turn;
  rotate: z 1.57rad;

  /* Vector plus angle value */
  rotate: 1 1 1 90deg;
}
```

<!-- scroll-timeline ---------------------------------------------------------------------------------------------------------------------------->

# scroll-timeline = scroll-timeline-name +

```scss
 {
  //
}
```

<!-- scroll-snap-type ----------------------------------------------------------------------------------------------------------------------->

# scroll-snap-type

определяет строгость привязки

```scss
.scroll-snap-type {
  scroll-snap-type: none;
  scroll-snap-type: x; // Прокрутка контейнера привязывается только по горизонтальной оси.
  scroll-snap-type: y; // Прокрутка контейнера привязывается только по вертикальной оси.
  scroll-snap-type: block; // Прокрутка контейнера привязывается только по блоковой оси.
  scroll-snap-type: inline; // Прокрутка контейнера привязывается только по строчной оси
  scroll-snap-type: both; // Прокрутка контейнера независимо привязывается только по обоим осям (потенциально может привязываться к разным элементам на разных осях).
}
```

<!-- scrollbar-color  ---------------------------------------------------------------------------------------------------------------------->

# scrollbar-color

Цвет полосы прокрутки

```scss
 {
  // первое значение - полоса прокрутки, второе - ползунок
  scrollbar-color: rebeccapurple green;
}
```

# shape-outside

<!-- shape-outside --------------------------------------------------------------------------------------------------------------------------->

Позволяет сделать обтекание во float по определенной фигуре

```scss
 {
  shape-outside: circle(50%);
}
```

<!--scroll-timeline------------------------------------------------------------------------------------------------------------------------->

# scroll-timeline

```scss
 {
  //scroll-timeline-name  scroll-timeline-axis
  scroll-timeline: --custom_name_for_timeline block;
  scroll-timeline: --custom_name_for_timeline inline;
  scroll-timeline: --custom_name_for_timeline y;
  scroll-timeline: --custom_name_for_timeline x;
  scroll-timeline: none block;
  scroll-timeline: none inline;
  scroll-timeline: none y;
  scroll-timeline: none x;
}
```

<!-- table-layout ---------------------------------------------------------------------------------------------------------------------------->

# table-layout

позволяет управлять расположением элементов в таблице

```scss
.table-layout {
  table-layout: "fixed"; //не будет адаптировать
  table-layout: "auto"; //будет адаптировать таблицу под контент, а именно растягивать ячейки
}
```

<!-- text-align ---------------------------------------------------------------------------------------------------------------------------->

# text-align

CSS-свойство описывает, как линейное содержимое, наподобие текста, выравнивается в блоке его родительского элемента. text-align не контролирует выравнивание элементов самого блока, но только их линейное содержимое.

```scss
.text-align {
  text-align: left;
  text-align: right;
  text-align: center;
  text-align: justify;
  text-align: start;
  text-align: end;
  text-align: match-parent; //c учетом direction
  text-align: start end;
  text-align: "."; // до символа
  text-align: start ".";
  text-align: "." end;
}
```

<!-- text-align-last------------------------------------------------------------------------------------------------------------------------>

# text-align-last

ак выравнивается последняя строка в блоке или строка, идущая сразу перед принудительным разрывом строки.

```scss
.text-align-last {
  text-align-last: auto;
  text-align-last: start;
  text-align-last: end;
  text-align-last: left;
  text-align-last: right;
  text-align-last: center;
  text-align-last: justify;
}
```

<!-- text --------------------------------------------------------------------------------------------------------------------->

# text-decoration

Декорирование текста

text-decoration = ext-decoration-color + text-decoration-line + text-decoration-style + text-decoration-thickness

```scss
 {
  //декорирование текста
  text-decoration-line: underline | overline | line-through | blink; //где находится линия
  text-decoration-style: solid | double | dotted | dashed | wavy; //цвет линии
  text-decoration-line: underline overline; // может быть две
  text-decoration-line: overline underline line-through;

  // цвет знака ударения
  text-emphasis-color: currentColor;
}
```

<!-- text-decoration-color ----------------------------------------------------------------------------------------------------------------->

# text-decoration-color

Определяет цвет подчеркивания

```scss
 {
  // шорткат для text-decoration-line, text-decoration-style, ext-decoration-color
  text-decoration: line-through red wavy;
  text-decoration-color: red;
}
```

<!-- text-decoration-skip ----------------------------------------------------------------------------------------------------------------->

# text-decoration-skip

при добавлении подчеркивания сделать сплошную линию, либо с прерыванием на буквы у,р,д

```scss
 {
  text-decoration-skip-ink: auto | none;
}
```

<!-- text-decoration-thickness ------------------------------------------------------------------------------------------------------------->

# text-decoration-thickness

Ширина линии подчеркивания

```scss
 {
  text-decoration-thickness: 0.1em;
  text-decoration-thickness: 3px;
}
```

<!-- text-emphasis-------------------------------------------------------------------------------------------------------------------------->

# text-emphasis

Добавит элементы поверх текста

text-emphasis = text-emphasis-position + text-emphasis-style + text-emphasis-color.

```scss
 {
  text-emphasis: "x";
  text-emphasis: "点";
  text-emphasis: "\25B2";
  text-emphasis: "*" #555;
  text-emphasis: "foo"; /* Should NOT use. It may be computed to or rendered as 'f' only */

  /* Keywords value */
  text-emphasis: filled;
  text-emphasis: open;
  text-emphasis: filled sesame;
  text-emphasis: open sesame;

  // возможные значения
  //  dot | circle | double-circle | triangle | sesame

  /* Keywords value combined with a color */
  text-emphasis: filled sesame #555;
}
```

<!-- text-emphasis-color ---------------------------------------------------------------------------------------------------------------------------->

# text-emphasis-color

Цвет элементов над буквами

```scss
 {
  text-emphasis-color: #555;
  text-emphasis-color: blue;
  text-emphasis-color: rgb(90 200 160 / 80%);
}
```

<!-- text-emphasis-position  --------------------------------------------------------------------------------------------------------------->

# text-emphasis-position

расположение элементов над буквами

```scss
text-emphasis-position. {
  text-emphasis-position: auto;

  /* Keyword values */
  text-emphasis-position: over;
  text-emphasis-position: under;

  text-emphasis-position: over right;
  text-emphasis-position: over left;
  text-emphasis-position: under right;
  text-emphasis-position: under left;

  text-emphasis-position: left over;
  text-emphasis-position: right over;
  text-emphasis-position: right under;
  text-emphasis-position: left under;
}
```

<!-- text-emphasis-style ------------------------------------------------------------------------------------------------------------------->

# text-emphasis-style

элемент для вставки

```scss
.text-emphasis-style {
  text-emphasis-style: "x";
  text-emphasis-style: "\25B2";
  text-emphasis-style: "*";

  /* Keyword values */
  text-emphasis-style: filled;
  text-emphasis-style: open;
  text-emphasis-style: dot;
  text-emphasis-style: circle;
  text-emphasis-style: double-circle;
  text-emphasis-style: triangle;
  text-emphasis-style: filled sesame;
  text-emphasis-style: open sesame;
}
```

<!-- text-overflow ------------------------------------------------------------------------------------------------------------------------->

# text-overflow

```scss
 {
  // обрежет текст
  text-overflow: clip;
  // поставит троеточие (два значения для rtl)
  text-overflow: ellipsis ellipsis;
  text-overflow: ellipsis " [..]";
  text-overflow: ellipsis "[..] ";
}
```

<!-- text-shadow ---------------------------------------------------------------------------------------------------------------------------->

# text-shadow

тень от текста

```scss
 {
  /* смещение-x | смещение-y | радиус-размытия | цвет */
  text-shadow: 1px 1px 2px black;

  /* цвет | смещение-x | смещение-y | радиус-размытия */
  text-shadow: #fc0 1px 0 10px;

  /* смещение-x | смещение-y | цвет */
  text-shadow: 5px 5px #558abb;

  /* цвет | смещение-x | смещение-y */
  text-shadow: white 2px 5px;

  /* смещение-x | смещение-y
/* Используем значения по умолчанию для цвета и радиуса-размытия */
  text-shadow: 5px 10px;
}
```

<!-- text-transform ------------------------------------------------------------------------------------------------------------------------>

# text-transform

Преобразует текст

```scss
 {
  text-transform: none;
  text-transform: capitalize;
  text-transform: uppercase;
  text-transform: lowercase;
  text-transform: full-width;
  text-transform: full-size-kana;
  text-transform: math-auto;
}
```

<!-- text-wrap  ---------------------------------------------------------------------------------------------------------------------------->

# text-wrap

Контролирует перенос текста внутри блока

```scss
.text-wrap {
  text-wrap: wrap; //обычный перенос при переполнение
  text-wrap: nowrap; //отмена переноса
  text-wrap: balance; //лучшее соотношение в плане длины строк
  text-wrap: pretty; // более медленный алгоритм wrap
  text-wrap: stable;
}
```

<!-- top-right-bottom-left------------------------------------------------------------------------------------------------------------------>

# top-right-bottom-left

Позиционирование для position:absolute | relative | sticky. Если заданы height: auto | 100% то будут учитываться оба

```scss
 {
  //
}
```

<!-- transform  ---------------------------------------------------------------------------------------------------------------------------->

# transform

Позволяет растягивать, поворачивать, масштабировать элемент

```scss
.transform {
  transform: none;

  transform: matrix(1, 2, 3, 4, 5, 6);
  transform: matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  transform: perspective(17px);
  transform: rotate(0.5turn);
  transform: rotate3d(1, 2, 3, 10deg);
  transform: rotateX(10deg);
  transform: rotateY(10deg);
  transform: rotateZ(10deg);
  transform: translate(12px, 50%);
  transform: translate3d(12px, 50%, 3em);
  transform: translateX(2em);
  transform: translateY(3in);
  transform: translateZ(2px);
  transform: scale(2, 0.5);
  transform: scale3d(2.5, 1.2, 0.3);
  transform: scaleX(2);
  transform: scaleY(0.5);
  transform: scaleZ(0.3);
  transform: skew(30deg, 20deg);
  transform: skewX(30deg);
  transform: skewY(1.07rad);

  /* Мультифункциональные значения */
  transform: translateX(10px) rotate(10deg) translateY(5px);
  transform: perspective(500px) translate(10px, 0, 20px) rotateY(3deg);
}
```

Если свойство имеет значение, отличное от none, будет создан контекст наложения. В этом случае, элемент будет действовать как содержащий блок для любых элементов position: fixed; или position: absolute; которые он содержит.

Свойство неприменимо: неизменяемые инлайновые блоки, блоки таблица-колонка, и блоки таблица-колонка-группа

<!--transform-box--------------------------------------------------------------------------------------------------------------------------->

# transform-box

определяет к чему будет приниматься трансформация

```scss
.transform-box {
  transform-box: content-box; //Поле содержимого
  transform-box: border-box; //пограничный блок
  transform-box: fill-box; //Ограничивающий блок
  transform-box: stroke-box; //Ограничивающий контур штриха
  transform-box: view-box; //Ближайший вьюпорт SVG
}
```

<!--transform-origin ----------------------------------------------------------------------------------------------------------------------->

# transform-origin

Относительно какой точки будет применяться трансформация

```scss
 {
  transform-origin: 2px;
  transform-origin: bottom;

  /* x-offset | y-offset */
  transform-origin: 3cm 2px;

  /* x-offset-keyword | y-offset */
  transform-origin: left 2px;

  /* x-offset-keyword | y-offset-keyword */
  transform-origin: right top;

  /* y-offset-keyword | x-offset-keyword */
  transform-origin: top right;

  /* x-offset | y-offset | z-offset */
  transform-origin: 2px 30% 10px;

  /* x-offset-keyword | y-offset | z-offset */
  transform-origin: left 5px -3px;

  /* x-offset-keyword | y-offset-keyword | z-offset */
  transform-origin: right bottom 2cm;

  /* y-offset-keyword | x-offset-keyword | z-offset */
  transform-origin: bottom right 2cm;
}
```

<!-- transition ---------------------------------------------------------------------------------------------------------------------------->

# transition

transition - укороченная запись для transition-property, transition-duration, transition-timing-function, и transition-delay

Значения по умолчанию:
transition-delay: 0s
transition-duration: 0s
transition-property: all
transition-timing-function: ease
transition-behavior: normal

```scss
.transition {
  transition: margin-left 4s;
  /* имя свойства | длительность | задержка */
  transition: margin-left 4s 1s;
  /* имя свойства | длительность | временная функция | задержка */
  transition: margin-left 4s ease-in-out 1s;
  /* Применить к 2 свойствам */
  transition: margin-left 4s, color 1s;
  /* Применить ко всем изменённым свойствам */
  transition: all 0.5s ease-out;
}
```

Объединение нескольких анимаций

```css
.elementToTransition {
  /* что анимировать all – все элементы */
  transition-property: background-color, border-color;
  /* длительность анимации */
  transition-duration: 1s 2s;
  /* временная функция анимации */
  transition-timing-function: cubic-bezier() ease;
  /* Задержка анимации */
  transition-delay: 2s 0.3s;
  /* все вместе */
  transition: background-color 1s as ease 2ms, border-color 2s ease;
}
```

<!-- transition-behavior ------------------------------------------------------------------------------------------------------------------->

# transition-behavior

Позволяет запускать анимацию на дискретных свойствах. Так как анимация будет до 50% и после. Исключение display:none и visibility:hidden

```scss
 {
  transition-behavior: allow-discrete; //позволяется анимировать
  transition-behavior: normal;
}
```

<!--transition-timing-function  ------------------------------------------------------------------------------------------------------------>

# transition-timing-function

```scss
.transition-timing-function {
  transition-timing-function: ease; //cubic-bezier(0.25, 0.1, 0.25, 1.0)
  transition-timing-function: linear; //cubic-bezier(0.0, 0.0, 1.0, 1.0)
  transition-timing-function: ease-in; //cubic-bezier(0.42, 0, 1.0, 1.0)
  transition-timing-function: ease-out; //cubic-bezier(0, 0, 0.58, 1.0)
  transition-timing-function: ease-in-out; //cubic-bezier(0.42, 0, 0.58, 1.0)
  transition-timing-function: cubic-bezier(p1, p2, p3, p4);
  // дискретные функции
  transition-timing-function: steps(n, jump-start);
  transition-timing-function: steps(n, jump-end);
  transition-timing-function: steps(n, jump-none);
  transition-timing-function: steps(n, jump-both);
  transition-timing-function: steps(n, start);
  transition-timing-function: steps(n, step-start); //jump-start.
  transition-timing-function: steps(n, step-end); // jump-end.
  transition-timing-function: step-start; //steps(1, jump-start)
  transition-timing-function: step-end; // steps(1, jump-end)
}
```

<!-- transform-style ----------------------------------------------------------------------------------------------------------------------->

# transform-style

Позиционирование 3d элементов

```scss
.transform-style {
  transform-style: preserve-3d; // Показывает, что дочерний элемент должен быть спозиционирован в 3D-пространстве.
  transform-style: flat; // Показывает, что дочерний элемент лежит в той же плоскости, что и родительский.
}
```

<!-- user-select --------------------------------------------------------------------------------------------------------------------------->

# user-select

Отвечает за возможность выделять текст

```scss
.user-select {
  user-select: none;
  user-select: auto;
  user-select: text;
  user-select: contain;
  user-select: all;
}
```

<!-- vertical-align ------------------------------------------------------------------------------------------------------------------------>

# vertical-align

Позволяет вертикально выравнять inline или inline-block элемент (нужно применять к элементу, который нужно выровнять) может использоваться в таблицах

```scss
 {
  vertical-align: baseline;
  vertical-align: sub;
  vertical-align: super;
  vertical-align: text-top;
  vertical-align: text-bottom;
  vertical-align: middle;
  vertical-align: top;
  vertical-align: bottom;
}
```

<!-- view-timeline ------------------------------------------------------------------------------------------------------------------------->

# view-timeline = view-timeline-name + view-timeline-axis

Определяет временную шкалу для анимации от видимости элемента

```scss
 {
  view-timeline: --custom_name_for_timeline block;
  view-timeline: --custom_name_for_timeline inline;
  view-timeline: --custom_name_for_timeline y;
  view-timeline: --custom_name_for_timeline x;
  view-timeline: none block;
  view-timeline: none inline;
  view-timeline: none y;
  view-timeline: none x;

  //view-timeline-name значения
  view-timeline-name: none;
  view-timeline-name: --custom_name_for_timeline;

  //view-timeline-axis значения
  view-timeline-axis: block;
  view-timeline-axis: inline;
  view-timeline-axis: y;
  view-timeline-axis: x;
}
```

<!-- view-timeline-inset ------------------------------------------------------------------------------------------------------------------->

# view-timeline-inset

Корректирует срабатывание анимации относительно скролла

Если значение положительное, положение начала/конца анимации будет перемещено внутри области прокрутки на указанную длину или процент.
Если значение отрицательное, то позиция начала/конца анимации будет перемещена за пределы области прокрутки на указанную длину или процент, т. е. анимация начнется до того, как появится в области прокрутки, или закончится после того, как анимация покинет область прокрутки.

```scss
.view-timeline-inset {
  //* Single value */
  view-timeline-inset: auto;
  view-timeline-inset: 200px;
  view-timeline-inset: 20%;

  /* Two values */
  view-timeline-inset: 20% auto;
  view-timeline-inset: auto 200px;
  view-timeline-inset: 20% 200px;
}
```

<!-- white-space ---------------------------------------------------------------------------------------------------------------------------->

# white-space

Свойство white-space управляет тем, как обрабатываются пробельные символы внутри элемента.

```scss
 {
  white-space: normal; //Последовательности пробелов объединяются в один пробел.
  white-space: nowrap; //не переносит строки (оборачивание текста) внутри текста.
  white-space: pre; //Последовательности пробелов сохраняются так, как они указаны в источнике.
  white-space: pre-wrap; //как и в pre + <br/>
  white-space: pre-line; //только <br />
  white-space: break-spaces;
}
```

<!-- width --------------------------------------------------------------------------------------------------------------------------------->

# width

```scss
 {
  // Ширина - фиксированная величина.
  width: 3.5em;
  width: anchor-size(width);
  width: calc(anchor-size(--myAnchor self-block, 250px) + 2em);

  width: 75%; // Ширина в процентах - размер относительно ширины родительского блока.

  /* Keyword values */
  width: none;
  width: max-content;
  width: min-content;
  width: fit-content;
  width: fit-content(20em);
  width: stretch;
}
```

<!--  word-break---------------------------------------------------------------------------------------------------------------------------->

# word-break

Где будет установлен перевод на новую строку

```scss
.word-break {
  word-break: normal;
  word-break: break-all;
  word-break: keep-all;
  word-break: break-word;
}
```

<!-- word-spacing ------------------------------------------------------------------------------------------------------------------------>

# word-spacing

расстояние между словами

```scss
 {
  word-spacing: "px", "%";
}
```

<!-- word-wrap --------------------------------------------------------------------------------------------------------------------------->

# word-break

```scss
 {
  word-wrap: "normal" | "break-word" | "inherit"; //перенос строки при переполнении
}
```

<!-- writing-mode ---------------------------------------------------------------------------------------------------------------------------->

# writing-mode

Позволяет перевернуть блок с текстом текст

```scss
 {
  writing-mode: horizontal-tb; // поток - сверху вниз, предложения - слева направо
  writing-mode: vertical-rl; // поток - справа налево, предложения - вертикально
  writing-mode: vertical-lr; // поток - слева направо, предложения - вертикально
}
```

<!-- webkit-background-clip ---------------------------------------------------------------------------------------------------------------------------->

# webkit-background-clip

Обрезка фона под текст

```css
.text-clip {
  -webkit-background-clip: text;
}
```

<!-- webkit-text-fill-color ---------------------------------------------------------------------------------------------------------------------------->

# webkit-text-fill-color

?Заливка текста

```css
.text-clip {
  -webkit-text-fill-color: transparent;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# moz-свойства

```scss
 {
  // учет рамок при вычислении высоты/ширины
  -moz-float-edge: content-box;
  -moz-float-edge: margin-box;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# -webkit- свойства

```scss
.webkit {
  // -webkit-border-before-color + -webkit-border-before-style + -webkit-border-before-width
  -webkit-border-beforenon-standard: ;
  -webkit-box-reflectnon-standard: ;
  // позволяет добавить троеточие
  -webkit-line-clamp: 3;
  -webkit-mask-box-imagenon-standard: ;
  -webkit-mask-compositenon-standard: ;
  -webkit-mask-position-xnon-standard: ;
  -webkit-mask-position-ynon-standard: ;
  -webkit-mask-repeat-xnon-standard: ;
  -webkit-mask-repeat-ynon-standard: ;
  -webkit-tap-highlight-colornon-standard: ;
  -webkit-text-fill-color: ;
  -webkit-text-securitynon-standard: ;
  -webkit-text-stroke: ;
}
```
