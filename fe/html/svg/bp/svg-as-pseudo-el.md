# SVG BP. Добавление SVG как псевдоэлемент

```scss
&__item::before {
  // (II) определяем содержание
  content: "";
  // (II) определяем размерность
  display: inline-block;
  height: 1rem;
  width: 1rem;
  margin-right: 0.7px;

  //(II) Для старых браузеров (не будет работать цвет)
  background-image: url(../img/chevron-thin-right.svg);
  background-size: cover;

  //(II) С использованием маски
  //(II) устанавливаем цвет заливки
  @supports (-webkit-mask-image: url()) or (mask-image: url()) {
    background-color: var(--color-primary);
    //(II) цвет пробивается через svg
    -webkit-mask-image: url(../img/chevron-thin-right.svg);
    -webkit-mask-size: cover;
    mask-size: cover;
    //если не убрать в хроме не будет цвета у svg
    background-image: none;
  }
}
```
