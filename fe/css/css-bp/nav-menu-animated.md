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

Источник курс Shmidtman, там float, проект OUTDOORS
