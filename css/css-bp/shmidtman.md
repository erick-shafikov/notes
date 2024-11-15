# OUTDOORS. FLOAT

## Анимированное навигационное меню

Меню навигации, которое открывается при нажатии на кнопку

```html
<div class="navigation">
  <!-- чекбокс для идентификации нажатия -->
  <input type="checkbox" class="navigation__checkbox" id="navi-toggle" />
  <!-- label в виде кнопки -->
  <label for="navi-toggle" class="navigation__button">
    <span class="navigation__icon">&nbsp;</span>
  </label>
  <!-- задний фон панели навигации -->
  <div class="navigation__background">&nbsp;</div>

  <!-- панель навигации -->
  <nav class="navigation__nav">
    <ul class="navigation__list">
      <li class="navigation__item">
        <a href="#" class="navigation__link"><span>01</span> About Natorus</a>
      </li>
      <!-- другие пункты меню -->
    </ul>
  </nav>
</div>
```

```scss
// (*) - создание закругленного фона
// (**) - создание анимированного закругленного фона

//<div class="navigation">
.navigation {
  // открытие и закрытие блока навигации осуществляется с помощью связки Checkbox(display: none) и label
  &__checkbox {
    display: none; //убираем чекбокс
  }

  //Кнопка <label for="navi-toggle" class="navigation__button">MENU</label>
  &__button {
    background-color: $color-white;

    //кнопка размером 7*7(*)
    height: 7rem;
    width: 7rem;
    //позиционирование фиксированное для нахождения элемента на одном и том же месте
    position: fixed;
    top: 6rem;
    right: 6rem;
    border-radius: 50%;
    z-index: 2000;
    box-shadow: 0 1rem 3rem rgba($color-black, 0.1);
    text-align: center;
    cursor: pointer;

    @include respond(tab-port) {
      top: 4rem;
      right: 4rem;
    }

    @include respond(phone) {
      top: 3.5rem;
      right: 3.5rem;
    }
  }

  &__background {
    //изначально фон размером 6*6 и находится под кнопкой (его не видно)(*)
    height: 6rem;
    width: 6rem;
    // такой же круглой формы
    border-radius: 50%;
    // с фиксированным позиционированием
    position: fixed;
    // находится под кнопкой
    top: 6.5rem;
    right: 6.5rem;
    //задний фон - круговой градиент
    background-image: radial-gradient(
      $color-primary-light,
      $color-primary-dark
    );
    //под кнопкой(*)
    z-index: 1000;

    transition: transform 0.8s cubic-bezier(0.68, -0.6, 0.32, 1.6);

    @include respond(tab-port) {
      top: 4.5rem;
      right: 4.5rem;
    }

    @include respond(phone) {
      top: 3.5rem;
      right: 3.5rem;
    }

    //увеличиваем фон в 80 раз(*) transform: scale(80) будет ниже &__checkbox:checked ~ &__background
  }

  &__nav {
    // <nav class="navigation__nav">
    height: 100vh;
    position: fixed;
    top: 0;
    left: 0;
    //навигация выше фона
    z-index: 1500;

    //делаем прозрачным при неактивном ЧБ и занимает 0%, если не поставить 0% то ссылки останутся на своем месте(*)
    opacity: 0;
    width: 0;
    transition: all 0.8s;
  }

  &__list {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    list-style: none;
    text-align: center;
    width: 100%;
  }

  &__item {
    margin: 1rem;
  }

  &__link {
    &:link,
    &:visited {
      display: inline-block;
      font-size: 3rem;
      font-weight: 300;
      padding: 1rem 2rem;
      color: $color-white;
      text-decoration: none;
      text-transform: uppercase;

      background-image: linear-gradient(
        120deg,
        transparent 0%,
        transparent 50%,
        $color-white 50%
      );
      // Изначально, белая часть градиента находится по середине блока, но 200% смещает границу правее на размер блока (**)
      background-size: 220%;
      transition: all 0.4s;

      span {
        margin-right: 1.5rem;
        display: inline-block;
      }
    }

    &:hover,
    &:active {
      // при наведении размер изображения градиента занимает весь блок (**)
      background-position: 100%;
      color: $color-primary;
      transform: translateX(1rem);
    }
  }
  // основной момент - ссылаемся на состояние чекбокса, если он checked тогда сосед &__background
  &__checkbox:checked ~ &__background {
    //увеличиваем фон в 80 раз если chb:active (*)
    transform: scale(80);
  }

  &__checkbox:checked ~ &__nav {
    //делаем непрозрачным при активном ЧБ и занимает 100%(*)
    opacity: 1;
    width: 100%;
  }

  // icon <span class="navigation__icon">&nbsp;</span>

  &__icon {
    position: relative;
    margin-top: 3.5rem;

    //3 линии
    &,
    &::before,
    &::after {
      width: 3rem;
      height: 2px;
      background-color: $color-grey-dark-3;
      display: inline-block;
    }

    &::before,
    &::after {
      content: "";
      position: absolute;
      left: 0;
      transition: all 0.2s;
    }

    &::before {
      top: -0.8rem;
    }
    &::after {
      top: 0.8rem;
    }
  }

  &__button:hover &__icon::before {
    top: -1rem;
  }

  &__button:hover &__icon::after {
    top: 1rem;
  }

  &__checkbox:checked + &__button &__icon {
    background-color: transparent;
  }

  &__checkbox:checked + &__button &__icon::before {
    top: 0;
    transform: rotate(135deg);
  }

  &__checkbox:checked + &__button &__icon::after {
    top: 0;
    transform: rotate(-135deg);
  }
}
```

## Переворачивающаяся карточка при наведении

```html
<div class="card">
  <div class="card__side card__side--front">
    <div class="card__picture card__picture--1">&nbsp;</div>
    <h4 class="card__heading">
      <span class="card__heading-span card__heading-span--1"
        >The sea <br />Explore</span
      >
    </h4>
    <div class="card__details">
      <ul>
        <li>3 day tour</li>
        <li>Up to 30 people</li>
        <li>2 tour guides</li>
        <li>Sleep in cazy hotel</li>
        <li>Difficulty: easy</li>
      </ul>
    </div>
  </div>
</div>
```

```scss
//реализация перевернутой карточки
.card {
  //реализация перспективы при повороте
  perspective: 150rem;
  -moz-perspective: 150rem;
  //сохраняем место для смещения
  position: relative;
  //так как убираем из потока карты, то контейнер card теряет размерность, сделаем height как у карточки
  height: 52rem;

  &__side {
    height: 52rem;
    transition: all 0.8s ease;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    //не будет показывать перевернутую часть карты
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    border-radius: 3px;
    box-shadow: 0 1.5rem 4rem rgba($color-black, 0.15);
    overflow: hidden;

    &--front {
      //модификатор для исходного положения
      background-color: $color-white;
    }

    &--back {
      //модификатор для конечного положения
      //изначальное положение второй карты - перевернутое (*)
      transform: rotateY(180deg);

      &-1 {
        background-image: linear-gradient(
          to right bottom,
          $color-secondary-light,
          $color-secondary-dark
        );
      }

      &-2 {
        background-image: linear-gradient(
          to right bottom,
          $color-primary-light,
          $color-primary-dark
        );
      }

      &-3 {
        background-image: linear-gradient(
          to right bottom,
          $color-tertiary-light,
          $color-tertiary-dark
        );
      }
    }
  }

  //применяем поворот только на изначальную сторону
  &:hover &__side--front {
    transform: rotateY(-180deg);
  }

  &:hover &__side--back {
    //конечно положение второй карты - НЕперевернутое (*)
    transform: rotateY(0);
  }

  &__picture {
    background-size: cover;
    height: 23rem;
    //смешивание двух картинок(**)
    background-blend-mode: screen;
    //Обрезаем картинку на карточке
    --webkit-clip-path: polygon(0 0, 100% 0, 100% 85%, 0 100%);
    clip-path: polygon(0 0, 100% 0, 100% 85%, 0 100%);
    border-bottom-left-radius: 3px;
    border-top-right-radius: 3px;

    &--1 {
      //компилируется в css изображения будут лежать на один уровень выше именно поэтому url('../img/nat-5.jpg')
      //смешиваем картинку и градиент(**)
      background-image: linear-gradient(
          to right bottom,
          $color-secondary-light,
          $color-secondary-dark
        ), url("../img/nat-5.jpg");
    }

    &--2 {
      background-image: linear-gradient(
          to right bottom,
          $color-primary-light,
          $color-primary-dark
        ), url("../img/nat-6.jpg");
    }

    &--3 {
      background-image: linear-gradient(
          to right bottom,
          $color-tertiary-light,
          $color-tertiary-dark
        ), url("../img/nat-7.jpg");
    }
  }

  &__heading {
    font-size: 2.5rem;
    font-weight: 300;
    text-transform: uppercase;
    text-align: right;
    color: $color-white;
    position: absolute;
    top: 12rem;
    right: 2rem;
    width: 75%;
  }

  &__heading-span {
    padding: 1rem 1.5rem;
    //применяет padding, который будет тянуться за каждой строчкой
    --webkit-box-decoration-break: clone;
    box-decoration-break: clone;
    &--1 {
      background-image: linear-gradient(
        to right bottom,
        rgba($color-secondary-light, 0.85),
        rgba($color-secondary-dark, 0.85)
      );
    }

    &--2 {
      background-image: linear-gradient(
        to right bottom,
        rgba($color-primary-light, 0.85),
        rgba($color-primary-dark, 0.85)
      );
    }

    &--3 {
      background-image: linear-gradient(
        to right bottom,
        rgba($color-tertiary-light, 0.85),
        rgba($color-tertiary-dark, 0.85)
      );
    }
  }

  &__details {
    padding: 3rem;

    ul {
      list-style: none;
      width: 80%;
      margin: 0 auto;
      li {
        text-align: center;
        font-size: 1.5rem;
        padding: 1rem;

        &:not(:last-child) {
          border-bottom: 1px solid $color-grey-light-2;
        }
      }
    }
  }

  &__cta {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 90%;
    text-align: center;
  }

  &__price-box {
    text-align: center;
    color: $color-white;
    margin-bottom: 8rem;
  }

  &__price-only {
    font-size: 1.4rem;
    text-transform: uppercase;
  }

  &__price-value {
    font-size: 6rem;
    font-weight: 100;
  }

  //Поворот карточки товара невозможен на ipad так VP не подходит для мобильного
  // используем правило, что только для screen или нет свойства hover
  @media only screen and (max-width: 56.25px), only screen and (hover: none) {
    height: auto;
    border-radius: 3px;
    background-color: $color-white;
    box-shadow: 0 1.5rem 4rem rgba($color-black, 0.15);

    &__side {
      height: auto;
      position: relative;
      box-shadow: none;

      &--back {
        transform: rotateY(0);
        clip-path: polygon(0 15%, 100% 0%, 100% 100%, 0% 100%);
      }
    }

    &:hover &__side--front {
      transform: rotateY(0);
    }

    &__details {
      padding: 1rem 3rem;
    }
    // Call to action
    &__cta {
      position: relative;
      top: 0;
      left: 0;
      transform: translate(0);
      width: 100%;
      padding: 7rem 4rem 4rem 4rem;
    }

    &__price-box {
      margin-bottom: 3rem;
    }

    &__price-value {
      font-size: 4rem;
    }
  }
}
```

## Модальный popup

```html
<!-- кнопка активации -->
<a href="#popup" class="btn btn--white">Book now!</a>
<!-- popup -->
<div class="popup" id="popup">
  <div class="popup__content">
    <div class="popup__left">
      <img src="img/nat-8.jpg" alt="Tour profile" class="popup__img" />
      <img src="img/nat-9.jpg" alt="Tour profile" class="popup__img" />
    </div>
    <div class="popup__right">
      <!-- при клике на крстик, прокрутка осуществится до элемента с id === section-tours (***)  -->
      <a href="#section-tours" class="popup__close">&times;</a>
      <h2 class="heading-secondary u-margin-bottom-medium">
        Start booking now
      </h2>
      <h3 class="heading-tertiary u-margin-bottom-small">
        Important &ndash; Please read these tearms before booking
      </h3>
      <p class="popup__text">
        <!-- текст -->
      </p>
      <a href="#" class="btn btn--green">Book now</a>
    </div>
  </div>
</div>
```

```scss
// * - разделение текстового блока на колонки
// ** - позиционированное элементов с помощью таблицы
// *** - обработка клика
// supports (IV)
.popup {
  height: 100vh;
  width: 100%;
  position: fixed;
  top: 0;
  left: 0;
  background-color: rgba($color-black, 0.8);
  z-index: 9999;
  //изначально элемент скрыт(***)
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s;
  overflow: auto;
  // если свойство поддерживается то применим их(IV)
  @supports (-webkit-backdrop-filter: blur(10px)) or
    (backdrop-filter: blur(10px)) {
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    background-color: rgba($color-black, 0.3);
  }

  &__content {
    //применяем миксин центрирования
    @include absCenter;
    width: 75%;
    background-color: $color-white;
    box-shadow: 0 2rem 4rem rgba($color-black, 0.2);
    border-radius: 3px;
    //Разделим с помощью таблицы(**)
    display: table;
    overflow: hidden;
    //изначально элемент скрыт(***)
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.25);
    transition: all 0.4s 0.2s;
  }

  &__left {
    //1/3 под левую секцию(**)
    width: 33.33333333%;
    display: table-cell;
    vertical-align: middle;

    @include respond(tab-land) {
      display: inline-block;
      width: 100%;
      margin: 0 auto;
    }
  }

  &__right {
    //2/3 под правую секцию(**)
    width: 66.6666667%;
    display: table-cell;
    vertical-align: middle;
    padding: 3rem 5rem;

    @include respond(tab-land) {
      display: inline-block;
      width: 100%;
    }
  }

  &__img {
    display: block;
    width: 100%;

    @include respond(tab-land) {
      display: inline;
      width: 45%;
      margin: 1rem 1rem;
      &:last-child {
        float: right;
      }
    }
  }

  &__text {
    font-size: 1.4rem;
    margin-bottom: 4rem;

    -moz-column-count: 2;
    -moz-column-gap: 4rem;
    -moz-column-rule: 1px solid $color-grey-light-2;
    // разделение текста на две колонки (*)
    column-count: 2;
    // размер промежутка между колонками (*)
    column-gap: 4rem;
    // разделитель (*)
    column-rule: 1px solid $color-grey-light-2;

    -moz-hyphens: auto;
    -ms-hyphens: auto;
    -webkit-hyphens: auto;
    //правильный разделитель слов (*)
    hyphens: auto;

    @include respond(tab-land) {
      -moz-column-count: 1;
      column-count: 1;
    }
  }

  // при добавлении в поисковую строку названия id элемента, добавляется класс target
  &:target {
    opacity: 1;
    visibility: visible;
  }

  &:target &__content {
    //становится не прозрачным(***)
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }

  //при клике на крестик, добавляется класс# к поисковой строке, значит popup не в состоянии target(***)
  &__close {
    &:link,
    &:visited {
      color: $color-grey-dark;
      position: absolute;
      top: 2.5rem;
      right: 2.5rem;
      font-size: 3rem;
      text-decoration: none;
      display: inline-block;
      transition: all 0.2s;
      line-height: 1;
    }

    &:hover {
      color: $color-primary;
    }
  }
}
```

# TRILLO. FLEX

## Боковое меню с выезжающим фоном для каждой опции меню

Реализован в виде списка ссылок с SVG элементами

```html
<nav class="sidebar">
  <ul class="side-nav">
    <!-- изначально выбираем элемент -->
    <li class="side-nav__item side-nav__item--active">
      <a href="#" class="side-nav__link">
        <svg class="side-nav__icon">
          <use xlink:href="img/sprite.svg#icon-home"></use>
        </svg>
        <span>Hotel</span>
      </a>
    </li>
  </ul>
</nav>
```

```scss
.side-nav {
  &__item {
    // li имеет relative позиционирование, что бы закрепить дочерний элемент
    position: relative;
  }

  // (I) добавляем псевдо-элемент в виде прямоугольника
  &__item::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 3px;
    background-color: var(--color-primary);
    // (I) изначально убран растягивание по вертикали
    transform: scaleY(0);
    // transform-origin для управления анимацией
    // 1-появляется сначала прямоугольник, потом расширяется width с задержкйо
    transition: transform 0.2s, width 0.4s cubic-bezier(1, 0, 0, 1) 0.2s,
      background-color 0.1s;
  }

  &__item:hover::before,
    // для изначально выбранного элемента без hover
  &__item--active::before {
    //для тех кто имеет модификатор active
    // (I) при наведении сначала расширяется по вертикали прямоугольник
    transform: scaleY(1);
    // (I) потом увеличивается на всю ширину
    width: 100%;
  }
  // заливается фоном
  &__item:active::before {
    background-color: var(--color-primary-light);
  }

  // убираем стиль у ссылок
  &__link:link,
  &__link:visited {
    color: var(--color-grey-light-1);
    text-decoration: none;
    text-transform: uppercase;
    display: block;
    padding: 1.5rem 3rem;
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
  }

  &__icon {
    width: 1.75rem;
    height: 1.75rem;
    margin-right: 2rem;
    // подтягивает цвет шрифтов и при разных состояниях (haver)
    fill: currentColor;

    @media only screen and (max-width: $bp-small) {
      margin-right: 0;
      margin-bottom: 0.7rem;
      width: 1.5rem;
      height: 1.5rem;
    }
  }
}
```

# NEXTER. GRID

# GRID сетка

```scss
.container {
  display: grid;
  grid-template-rows: 80vh min-content 40vw repeat(3, min-content);
  //разбиваем на 8 колонок (I)
  //первая колонка - для sidebar - 8rem
  // втора колонка в роли border - минимум 6rem максимум 1fr
  // c 3 по 7 - колонки растягиваются автоматически или занимают 14rem 1400/8
  // последняя колонка, как вторая
  grid-template-columns:
    [sidebar-start] 8rem [sidebar-end full-start] minmax(6rem, 1fr)
    [center-start] repeat(8, [col-start] minmax(min-content, 14rem) [col-end])
    [center-end]
    minmax(6rem, 1fr)
    [full-end];

  @media only screen and (max-width: $bp-large) {
    // добавляем строку сверху (II)
    grid-template-rows: 6rem 80vh min-content 40vw repeat(3, min-content);
    grid-template-columns:
        // убираем side-bar на вверх меняя сетку (II)

      [full-start] minmax(6rem, 1fr)
      [center-start] repeat(8, [col-start] minmax(min-content, 14rem) [col-end])
      [center-end]
      minmax(6rem, 1fr)
      [full-end];
  }

  @media only screen and (max-width: $bp-medium) {
    // смещаем секцию с риелторами в отдельный ряд (III)
    // что бы heading занимал весь экран нужно вычесть 6rem (высота sidebar)
    grid-template-rows: 6rem calc(100vh - 6rem);
  }
}
```

Привязка к колонке

```scss
.features {
  // занимает центральную колонку с repeat (I)
  grid-column: center-start / center-end;
}
```
