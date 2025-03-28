Работа с svg

Дополнительные преимущества SVG:

- Текст в векторном изображении остаётся машинописным (то есть доступным для поисковика, что улучшает SEO).
- SVG легко поддаются стилизации/программированию (scripting), потому что каждый компонент изображения может быть - стилизован с помощью CSS или запрограммирован с помощью JavaScript.

Минусы

- SVG может очень быстро стать сложным в том смысле, что размер файла увеличивается; сложные SVG-изображения также создают большую вычислительную нагрузку на браузер.
- SVG может быть сложнее создать, нежели растровое изображение, в зависимости от того, какое изображение необходимо создать.
- не поддерживается старыми версиями браузеров, то есть не подойдёт для сайтов, поддерживающих Internet Explorer 8 или старее.

```html
<!-- пример круг -->
<svg
  version="1.1"
  baseProfile="full"
  width="300"
  height="200"
  xmlns="http://www.w3.org/2000/svg"
>
  <rect width="100%" height="100%" fill="black" />
  <circle cx="150" cy="100" r="90" fill="blue" />
</svg>
```

что бы добавить

```html
<img
  src="equilateral.svg"
  alt="triangle with all three sides equal"
  height="87px"
  width="100px"
/>
```

- svg нельзя управлять с помощью js если только svg не находится в общей разметке
- используют собственные css стили, нельзя определять стили внутри css файлов сайта

# Вставка svg-спрайта

```html
<!-- Вставка svg-спрайта-->
<svg class="feature__icon">
  <use xlink:href="img/sprite.svg#icon-global"></use>
</svg>
```

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

# currentColor

атрибут fill должен быть currentColor что бы цвета применялись
