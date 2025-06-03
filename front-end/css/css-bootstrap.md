# BOOTSTRAP

# display

```html
<!-- из блочного в строковый -->
<div class="d-inline">d-inline</div>
<!-- из строчного в блочный -->
<span class="d-block">d-block</span>
<!-- сокрытие элементов -->
<div class="d-none d-lg-block">скрыть на экранах меньше lg</div>
<!-- сокрытие элементов для печати -->
<div class="d-print-none">
  <!-- сокрытие элементов, но при печати в блочный -->
  <div class="d-none d-print-block"></div>
</div>
```

# flex (контейнеры)

```html
<!-- flex контейнер -->
<div class="d-flex"></div>
<!-- inline flex контейнер -->
<div class="d-inline-flex"></div>
<!-- inline flex контейнер с адаптивной сеткой -->
<div class=".d-sm-flex"></div>
<!-- направление, другие значения: flex-row-reverse, flex-column. -->
<div class="d-flex flex-row"></div>
flex-xl-row - для адаптивной сетки
<div class="d-flex justify-content-start"></div>
<!-- Выравнивание контента, justify-content-lg-end - адаптивный вариант -->
<div class="d-flex align-items-start"></div>
<!-- Горизонтальный вариант выравнивания -->
<div class="d-flex flex-nowrap"></div>
<!-- настройка переноса контента, flex-wrap, flex-wrap-reverse -->
<div class="d-flex align-content-stretch flex-wrap">...</div>
<!-- позиционирование контента по поперечной оси -->
```

# flex-элементы

```html
<div class="align-self-start"></div>
<!-- Выравнивание flex-элемента -->
<div class="d-flex"></div>
<div class="flex-fill"></div>
<!-- Элемент займет пространство по контенту -->
<div class="p-2 flex-grow-1"></div>
<!-- настройка жадности элемента -->
<div class="p-2 flex-shrink-1"></div>
<!-- настройка отдачи элемента -->
<div class="me-auto"></div>
<!-- откинуть следующие элементы  margin-right: auto -->
<div class="ms-auto p-2"></div>
<!-- margin-left: auto -->
<div class="order-3 p-2"> //порядок</div>
```

# container

```html
<!-- container класс имеет собственные padding и margin -->
<div class="container-sm">100% шириной до контрольной точки small</div>
<div class="container-md">100% шириной до контрольной точки medium</div>
<div class="container-lg">100% шириной до контрольной точки large</div>
<div class="container-xl">100% шириной до контрольной точки extra large</div>
<div class="container-xxl">
  100% шириной до контрольной точки extra extra large  
</div>
<!-- Пример адаптивных контейнеров -->
<div class="container-fluid"></div>
<!-- - займет все возможное место -->
```

# Layout-grid

Основные элементы col и row, row – определяет начало ряда, col – колонку в ряду, каждый ряд – это flex контейнер

<img src='./assets/css-bootstrap/layout-grid-1.png' height=80/>

```html
<div class="container">
  <div class="row">
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
  </div>
</div>
```

Неравное распределение размеров колонок (col-n)

<img src='./assets/css-bootstrap/layout-grid-2.png' height=80/>

```html
<div class="container">
  <div class="row">
    <div class="col">1 из 3</div>
    <div class="col-6">2 из 3 (шире)</div>
    <div class="col">3 из 3</div>
  </div>
</div>
```

Автоматическое определение ширины контента (md-auto)

<img src='./assets/css-bootstrap/layout-grid-3.png' height=80/>

```html
<div class="container">
  <div class="row justify-content-md-center">
    <!-- отцентрирует весь контент внутри контейнера -->
    <div class="col col-lg-2">1 из 3</div>
    <div class="col-md-auto">Переменная ширина контента</div>
    <!-- второй будет по величине содержимого -->
    <div class="col col-lg-2">3 из 3</div>
  </div>
</div>
```

Адаптивное распределение рядов, которая начнется с контрольной точки sm

```html
<div class="container">
  <div class="row">
    <div class="col-sm-8">…</div>
    <div class="col-sm-4">…</div>
    <div class="col-6 col-md-4">…</div>
    <!-- смешанное сочетание -->
  </div>
</div>
```

# ряды колонок (row-col)

```html
<div class="container">
  <div class="row row-cols-2">
    <!-- распределит в 2 колонки в 2 ряда -->
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
  </div>
</div>
```

Автоматический перенос

```html
<div class="container">
  <div class="row row-cols-2">
    <!-- распределит в 2 колонки -->
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <!-- автоматически перенесется в следующий ряд-->
    <div class="col">Колонка</div>
  </div>
</div>
```

Автоматическое определение пространства

```html
<div class="container">
  <div class="row row-cols-auto">
    <!-- автоматически распределятся по содержимому-->
    <div class="col">Колонка</div>
    …
  </div>
</div>
```

Неравное распределение

```html
<div class="container">
  <div class="row row-cols-4">
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col-6">Колонка</div>
    <!-- автоматически займет половину row -->
    <div class="col">Колонка</div>
    <!-- автоматически перенесется на следующий ряд -->
  </div>
</div>
```

Адаптивный вариант распределения

```html
<div class="container">
  <!-- //начиная с sm – 2 колонки… -->
  <div class="row row-cols-1 row-cols-sm-2 row-cols-md-4">
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
    <div class="col">Колонка</div>
  </div>
</div>
```

Вложенные сетки

```html
<div class="container">
  <div class="row">
    <div class="col-sm-3">Уровень 1: .col-sm-3</div>
    <div class="col-sm-9">
      <div class="row">
        <div class="col-8 col-sm-6">Уровень 2: .col-8 .col-sm-6</div>
        <div class="col-4 col-sm-6">Уровень 2: .col-4 .col-sm-6</div>
      </div>
    </div>
  </div>
</div>
```

# колонки (col)

Вертикальное выравнивание

```html
<!-- отцентрирует весть текст -->
<div class="container text-center">
  <!-- распределит к верху контент в контейнере, другие варианты -->
  <div class="row align-items-start justify-content-start">
    <!-- align-items-center, align-items-end и горизонтально распределит к старту -->
    <div class="col align-self-end">Одна из трех колонок</div>
    <!-- применит к конкретному контейнеру -->
    <div class="col order-5">Одна из трех колонок</div>
    <!-- изменит порядок order-last и order-last определят порядок как первого и последнего элемента-->
    <div class="col">Одна из трех колонок</div>
  </div>
</div>
```

Разделение на ряды с помощью горизонтального разделителя

```html
<div class="container">
  <div class="row">
    <div class="col-6 col-sm-3">.col-6 .col-sm-3</div>
    <div class="col-6 col-sm-3">.col-6 .col-sm-3</div>
    <div class="w-100"></div>
    <!-- разделит на два ряда, хоть и элементы находятся в одном ряду -->
    <div class="w-100 d-none d-md-block"></div>
    <!-- адаптивный вариант разделителя -->
    <div class="col-6 col-sm-3">.col-6 .col-sm-3</div>
    <div class="col-6 col-sm-3">.col-6 .col-sm-3</div>
  </div>
</div>
```

# смещение

Смещение контента по сетке

```html
<div class="container">
  <div class="row">
    <div class="col-md-4">.col-md-4</div>
    <div class="col-md-4 offset-md-4">.col-md-4 .offset-md-4</div>
    <!-- //сместиться на 4 относительного первого, прижмется к концу -->
  </div>
  <div class="row">
    <div class="col-md-3 offset-md-3">.col-md-3 .offset-md-3</div>
    <div class="col-md-3 offset-md-3">.col-md-3 .offset-md-3</div>
  </div>
  <div class="row">
    <!-- //сброс смещения -->
    <div class="col-md-6 offset-md-3 offset-lg-0">.col-md-6 .offset-md-3</div>
  </div>
</div>
```

Использование col вне row

```html
<!-- займет 1/3 часть контейнера -->
<div class="col-3">.col-3: width of 25%</div>
<div class="col-sm-9">.col-sm-9: width of 75% above sm breakpoint</div>
```

Использование с float’ами

```html
<div class="clearfix">
  <img src="..." class="col-md-6 float-md-end mb-3 ms-md-3" alt="..." />
</div>
```

# промежутки

```html
<!-- горизонтальный -->
<div class="row gx-5"></div>
<!-- вертикальный -->
<div class="row gy-5"></div>
<!-- и горизонтальный и вертикальный -->
<div class="row g-2"></div>
<!-- адаптивные промежутки -->
<div class="row g-lg-3"></div>
<!-- нет промежутков -->
<div class="row g-0 text-center"></div>
```

# float

```html
<!-- верстка на float -->
<div class="float-sm-end"></div>
```

# Таблицы

```html
<!-- адаптивная таблица, на sm появиться полоса прокрутки -->
<table class="table-responsive-sm"></table>
<!-- создает обычную таблицу -->
<table class="table"></table>
<!-- создает цветную -->
<table class="table-primary"> </table>
<!-- создает акцентированную таблицу (выделяя нечетные строки) -->
<table class="table table-striped"> </table>
<!-- создает акцентированную таблицу (выделяя нечетные столбцы) -->
<table class="table table-striped-columns"> </table>
<!-- создает темную таблицу -->
<table class="table table-dark"></table>
<!-- создает таблицу с наведением -->
<table class="table table-hover"></table>
<!-- создает таблицу с границами -->
<table class="table table-bordered"></table>
<!-- создает таблицу без границ -->
<table class="table table-borderdless"></table>
<!-- создает меньшего размера -->
<table class="table table-sm"></table>
<!-- вертикальное выравнивание контента -->
<table class="table align-middle"></table>
<!--caption переместит на верх (по умолчанию внизу) -->
<table class="table align-middle caption-top"></table>
<caption>Список пользователей</caption>
<!-- оформление заголовка таблицы -->
<thead class="table-light">
  <!-- создает цветной ряд -->
  <tr class="table-primary">
  <!-- создает активную строку -->
  <tr class="table-active">
  <!-- создает разделитель -->
  <tr class="table-group-divider">
   <th scope="col">#</th>
   <th scope="col">Имя</th>
   <th scope="col">Фамилия</th>
   <th scope="col">Обращение</th>
  </tr>
 </thead>
 <tbody>
   <tr>
    <th scope="row">1</th>
    <!-- выровняет вертикально контент -->
    <td class="align-top">Mark</td>
</tr>
</tbody>
</table>
```

# Цвета и прозрачность

```html
<!-- background color === primary, text === white  -->
<div class="p-3 mb-2 bg-primary text-white">.bg-primary</div>
<!-- //градиент -->
<div
  class="p-3 mb-2 bg-primary text-white"
  style="background-image:var(--bs-gradient)"
>
  <!-- //настройка прозрачности -->
  <div class="bg-success p-2" style="--bs-bg-opacity: 0.5">
    Это успешный фон с непрозрачностью 50%
  </div>

  <!-- обычный цвет -->
  <p class="text-primary">.text-primary</p>
  <!-- цвет с акцентом -->
  <p class="text-primary-emphasis">.text-primary-emphasis</p>
  <!-- с background -->
  <p class="text-warning bg-dark">.text-warning</p>
  <!-- настройка opacity -->
  <p class="text-black-50 bg-white">.text-black-50</p>
  <!-- настройка opacity непрозрачность -->
  <div class="text-primary text-opacity-75">
    <div class="opacity-75">...</div>
  </div>
</div>
```

# overflow

```html
<!-- значения для overflow: overflow-hidden, overflow-visible, overflow-scroll -->
<div class="overflow-auto"></div>
<!-- overflow-x-hidden, overflow-x-visible, overflow-x-scroll -->
<div class="overflow-x-auto"></div>
<!-- overflow-y-hidden, overflow-y-visible, overflow-y-scroll -->
<div class="overflow-y-auto"></div>
```

# тень

```html
<div class="shadow-none p-3">Нет тени</div>
<div class="shadow-sm p-3">Маленькая тень</div>
<div class="shadow p-3">Обычная тень</div>
<div class="shadow-lg p-3">Большая тень</div>
```

# position

```html
<!-- позиционирование: position-relative, position-absolute, position-fixed, position-sticky -->
<div class="position-static">
     
  <div class="position-relative">
    <!-- к верхнему левому -->
    <div class="position-absolute top-0 start-0"></div>
    <!-- к верхнему правому -->
    <div class="position-absolute top-0 end-0"></div>
    <!-- к центру -->
    <div class="position-absolute top-50 start-50"></div>
    <!--  -->
    <div class="position-absolute bottom-50 end-50"></div>
    <!-- к левому нижнему -->
    <div class="position-absolute bottom-0 start-0"></div>
    <!-- к правому нижнему -->
    <div class="position-absolute bottom-0 end-0"></div>
  </div>
</div>
```

# translate

```html
<!-- по горизонтальной оси -->
<div class="position-absolute top-0 start-50 translate-middle-x"></div>
<!-- по вертикальной оси     -->
<div class="position-absolute top-50 start-0 translate-middle-y"></div>
<!-- сместит элемент по своему центру translate(-50%, -50%) -->
<div class="position-absolute top-0 start-0 translate-middle"></div>
```

# границы, borders

```html
<!-- добавление границ, по отдельности border-top, border-end, border-bottom, border-start -->
<div class="border"></div>
<!-- удаление: border border-top-0, border-end-0, border-bottom-0, border-start-0"></ -->
<div class="border border-0"></div>
<!-- цвет -->
<div class="border border-primary"></div>
<!-- настройка прозрачности границы -->
<div class="border border-success p-2" style="--bs-border-opacity: 0.5"></div>
<!-- настройка ширины границы -->
<div class="border border-5"></div>
<!-- скругленные границы, варианты:  rounded-top, rounded-end, rounded-bottom, rounded-start, rounded-circle, rounded-pill -->
<div class="rounded">Это граница успеха с непрозрачностью 50%</div>
<!-- закругление 50% -->
<span class="rounded-5"> </span>
```

# текст

```html
Выделение текста
<!-- Этот абзац будет полностью выделен при нажатии пользователем. -->
<p class="user-select-all"></p>
<!-- Этот абзац имеет поведение выбора по умолчанию. -->
<p class="user-select-auto"></p>
<!-- Этот абзац не будет доступен для выбора при нажатии пользователем -->
<p class="user-select-none"></p>
```

# ссылки

```html
<!-- ссылка по которой нельзя перейти    -->
<a href="#" class="pe-none"></a>
<!-- обычная ссылка -->
<a href="#" class="pe-auto"></a>
<!-- прозрачность ссылки -->
<a class="link-opacity-10" href="#"></a>
<!-- прозрачность ссылки при наведении -->
<a class="link-opacity-10-hover" href="#"></a>
<!-- выбор цвета подчеркивания -->
<a href="#" class="link-underline-primary"></a>
<!-- смещение линии подчеркивания -->
<a class="link-offset-1" href="#"></a>
<!-- непрозрачность линии подчеркивания -->
<a class="link-underline link-underline-opacity-0" href="#"></a>
<!-- смещение при наведении и смена прозрачности -->
<a
  class="link-offset-2 link-offset-3-hover link-underline link-underline-opacity-0 link-underline-opacity-75-hover"
  href="#"
></a>
<!-- цвет ссылки -->
<a href="#" class="link-primary"></a>
```

# изображения

```html
<img src="..." class="object-fit-contain" alt="..." /> //
<img src="..." class="object-fit-cover" alt="..." />
<img src="..." class="object-fit-fill" alt="..." />
<img src="..." class="object-fit-scale" alt="..." />
<img src="..." class="object-fit-none" alt="..." />
<img src="..." class="object-fit-sm-contain" alt="..." />
```

# breakpoints

```css
/* Минимальная ширина: */
/* @media (min-width: 576px) { ... } */
@include media-breakpoint-up(sm) {
}

/* Максимальная ширина */
/* @media (max-width: 575.98px) { ... } */
@include media-breakpoint-down(sm) {
}

/* Для конкретного размера:  */
/* @media (min-width: 768px) and (max-width: 991.98px) { ... } */
@include media-breakpoint-only(xs) {
}

/* Для промежутков: , xl) { ... } */
/* @media (min-width: 768px) and (max-width: 1199.98px) { ... } */
@include media-breakpoint-between(md, xl) {
}
```
