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

Источник курс Shmidtman, там flex, проект trillo
