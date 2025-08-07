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

Источник курс Shmidtman, там float, проект OUTDOORS
