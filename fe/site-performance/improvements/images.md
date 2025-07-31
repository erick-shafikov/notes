# BP. Работа с изображениями

Оптимизация загрузки большого количества изображений

- Зарезервировать место для img с помощью width и height
- показать сначала в плохом качестве
- сжатие
- уменьшение размера

```html
<img
  src="pillow.jpg"
  width="640"
  height="360"
  alt="purple pillow with flower pattern"
/>
```

- Ленивая отрисовка
- использование селекторов srcset (позволяет указать какую картинку загружать в зависимости от VP) и size работает как медиа-запрос

```html
<picture>
  <source media="(min-width: 768px)" srcset="tablet_image.png" />
  <source media="(min-width: 1024px)" srcset="desktop_image.png" />
  <img src="mobile_image.png" alt="" />
</picture>
```

- ленивая загрузка img loading === lazy дял изображений второго порядка, только не для больших изображений, так как загрузится после всего

```html
<img
  src="pillow.jpg"
  width="640"
  height="360"
  alt="purple pillow with flower pattern"
  loading="”lazy”"
/>
```

- отложенные изображения для изображений второго

```html
<link
  rel="”preload”"
  as="”image”"
  href="”mobileBanner.png”"
  media="(max-width:768px)"
/>
<link
  rel="”preload”"
  as="”image”"
  href="”desktopBanner.png”"
  media="(min-width:1600px)"
/>
```

- Размытая заглушка с помощью background-image
- сжатие изображений
- форматы wbeP и avif

```html
<picture>
  <source srcset="image.avif" type="image/avif" />
  <source srcset="image.webp" type="image/webp" />
  <img src="image.jpg" alt="..." width="800" height="600" />>
</picture>
```

- выбор изображений в css

```scss
.example-block {
  background-image: url("/example_400x225.png");
}

.webp .example-block {
  background-image: url("/example_400x225.webp");
}
```

- [спрайты для мелких изображений](../../html/svg/bp/sprite.md)
- спрайты для png

```css
.icon {
  background: url("/sprite.png");
  height: 20px;
  width: 20px;
}
.icon-check {
  background-position: 20px 0px;
}
.icon-close {
  background-position: 40px 0px;
}
```
