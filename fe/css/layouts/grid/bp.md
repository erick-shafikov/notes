# Grid — лучшие практики (BP)

## BP. Auto Grid — адаптивные колонки без медиа-запросов

Intrinsic layout: браузер сам решает сколько колонок влезет. `min(400px, 100%)` предотвращает горизонтальный overflow на малых экранах.

```scss
// auto-fit: пустые треки схлопываются, элементы растягиваются
.auto-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr));
}
```

```scss
// auto-fill: пустые треки резервируются, элементы не растягиваются если их мало
.auto-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fill, minmax(min(400px, 100%), 1fr));
}
```

## BP. Центрирование с grid

```scss
// быстрый способ отцентрировать один дочерний элемент
.center {
  display: grid;
  place-items: center;
}
```

## BP. Липкий footer

```scss
// header | main (занимает свободное место) | footer
.wrapper {
  min-height: 100%;
  display: grid;
  grid-template-rows: auto 1fr auto;
}
```

## BP. Текст поверх картинки

```html
<div class="container">
  <div class="image"></div>
  <div class="text">Текст</div>
</div>
```

```scss
.container {
  display: grid;
  justify-items: center;
}

// оба элемента помещаются в одну ячейку (1/1 → 1/1)
.image {
  grid-column: 1 / -1;
  grid-row: 1 / -1;
  width: 100px;
  height: 100px;
}

.text {
  grid-column: 1 / -1;
  grid-row: 1 / -1; // та же ячейка — накладывается поверх картинки
  align-self: center;
}
```

## BP. Карточка медиа-объект (картинка + текст)

```html
<div class="media">
  <div class="img"><img src="..." alt="..." /></div>
  <div class="content"><p>...</p></div>
  <div class="footer">Footer</div>
</div>
```

```scss
@media (min-width: 500px) {
  .media {
    display: grid;
    // колонка под картинку растёт по содержимому, но не более 200px
    grid-template-columns: fit-content(200px) 1fr;
    grid-template-rows: 1fr auto;
    grid-template-areas:
      "image content"
      "image footer";
    gap: 20px;
  }

  // перевёрнутая карточка (картинка справа)
  .media-flip {
    grid-template-columns: 1fr fit-content(250px);
    grid-template-areas:
      "content image"
      "footer  image";
  }

  .img     { grid-area: image; }
  .content { grid-area: content; }
  .footer  { grid-area: footer; }
}
```

## BP. Адаптивная сетка (mobile-first)

```html
<div class="wrapper">
  <header>Header</header>
  <article>Content</article>
  <aside>Sidebar</aside>
</div>
```

```scss
.wrapper {
  display: grid;
  gap: 10px;
  // мобильная версия: одна колонка, элементы в порядке DOM
}

@media (min-width: 767px) {
  .header  { grid-column: 1 / 3; grid-row: 1 / 2; }
  .article { grid-column: 1 / 2; grid-row: 2 / 3; }
  .aside   { grid-column: 2 / 3; grid-row: 2 / 3; }
}

@media (min-width: 1024px) {
  // меняем местами article и aside без изменения DOM
  .article { grid-column: 2 / 3; grid-row: 2 / 3; }
  .aside   { grid-column: 1 / 2; grid-row: 2 / 3; }
}
```

## BP. Трёхколоночный layout с областями

```html
<div class="wrapper">
  <header class="main-head">Header</header>
  <nav class="main-nav">Nav</nav>
  <article class="content">Content</article>
  <aside class="side">Sidebar</aside>
  <div class="ad">Advertising</div>
  <footer class="main-footer">Footer</footer>
</div>
```

```scss
.main-head   { grid-area: header; }
.content     { grid-area: content; }
.main-nav    { grid-area: nav; }
.side        { grid-area: sidebar; }
.ad          { grid-area: ad; }
.main-footer { grid-area: footer; }

// мобильная версия: одна колонка
.wrapper {
  display: grid;
  gap: 20px;
  grid-template-areas:
    "header"
    "nav"
    "content"
    "sidebar"
    "ad"
    "footer";
}

@media (min-width: 500px) {
  .wrapper {
    grid-template-columns: 1fr 3fr;
    grid-template-areas:
      "header  header"
      "nav     nav"
      "sidebar content"
      "ad      footer";
  }

  nav ul { display: flex; justify-content: space-between; }
}

@media (min-width: 700px) {
  .wrapper {
    grid-template-columns: 1fr 4fr 1fr;
    grid-template-areas:
      "header header  header"
      "nav    content sidebar"
      "nav    content ad"
      "footer footer  footer";
  }

  nav ul { flex-direction: column; }
}
```

## BP. 12-колоночная сетка

```html
<div class="wrapper">
  <div class="item1">Span 3</div>
  <div class="item2">Span 4, 2 rows</div>
  <div class="item3">Span 2</div>
  <div class="item4">Span to end</div>
</div>
```

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr); // 12 именованных колонок
  gap: 20px;
}

.item1 { grid-column: col-start / span 3; }           // с 1-й, занять 3
.item2 { grid-column: col-start 6 / span 4; grid-row: 1 / 3; } // с 6-й, 4 колонки, 2 ряда
.item3 { grid-column: col-start 2 / span 2; grid-row: 2; }
.item4 { grid-column: col-start 3 / -1; grid-row: 3; } // с 3-й до конца
```

Responsive 12-колоночный с переопределением:

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr);
  gap: 20px;
}

// на мобиле — всё на всю ширину
.wrapper > * { grid-column: col-start / span 12; }

@media (min-width: 500px) {
  .side          { grid-column: col-start / span 3; grid-row: 3; }
  .ad            { grid-column: col-start / span 3; grid-row: 4; }
  .content,
  .main-footer   { grid-column: col-start 4 / span 9; }
}

@media (min-width: 700px) {
  .main-nav  { grid-column: col-start / span 2; grid-row: 2 / 4; }
  .content   { grid-column: col-start 3 / span 8; grid-row: 2 / 4; }
  .side      { grid-column: col-start 11 / span 2; grid-row: 2; }
  .ad        { grid-column: col-start 11 / span 2; grid-row: 3; }
  .main-footer { grid-column: col-start / span 12; }
}
```

## BP. Список карточек с dense-заполнением

```html
<ul class="listing">
  <li>...</li>
  <li class="wide">...</li> <!-- широкая карточка -->
</ul>
```

```scss
.listing {
  list-style: none;
  display: grid;
  gap: 20px;
  // авто: сколько поместится колонок шириной от 200px
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
}

// внутренности карточки — flex для sticky CTA
.listing li {
  border: 1px solid #ffe066;
  border-radius: 5px;
  display: flex;
  flex-direction: column;
}

.listing .cta  { margin-top: auto; border-top: 1px solid #ffe066; padding: 10px; text-align: center; }
.listing .body { padding: 10px; }
```

Вариант с dense и широкими карточками:

```scss
.listing {
  display: grid;
  gap: 20px;
  grid-auto-flow: dense;           // заполняет дыры меньшими карточками
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
}

.listing .wide {
  grid-column-end: span 2; // широкая карточка занимает 2 колонки
}
```

## BP. Абсолютное позиционирование в grid

Grid-контейнер можно использовать как positioning context:

```scss
.wrapper {
  display: grid;
  position: relative; // контейнер для абсолютно позиционированных потомков
}
```

## BP. minmax + auto-flow

```scss
// flex альтернативы нет — flex делит свободное место, не задаёт min-size
.grid-container {
  display: grid;
  // авто-колонки с min-content минимумом и fr максимумом
  grid: none / auto-flow minmax(min-content, 1fr);
}
```
