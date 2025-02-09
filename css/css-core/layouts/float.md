<!-- Сетки. Float ---------------------------------------------------------------------------------------------------------------------------->

# Сетки. Float

Помимо Float компоновки существуют: Обычная (Normal Flow), Табличная,Float layout, Позиционированная, Множественные столбцы, Flex box, Сеточная

свойство позволяет настроить обтекание текста. Делает float-элемент display: block
после float элементов нужно вставить пустой элемент со свойством clear: both; что бы удалить float
Свойство float CSS указывает, что элемент должен быть взят из нормального потока и помещён вдоль левой или правой стороны его контейнера, где текст и встроенные элементы будут обтекать его

Добавить float

```css
.box {
  float: left;
  float: right;
  float: none;
}
```

```html
<h1>Simple float example</h1>

<!-- этот блок выйдет из потока, встанет слева, а параграф ниже займет пространство справа -->
<div class="box">Float</div>

<p>Lorem ipsum</p>

<p>orem ipsum</p>

<p>orem ipsum</p>
```

!!!добавления margin а первому параграфу после не даст желаемого результата, margin нужно добавлять к float элементу

# Обтекание по фигуре

Можно сделать с помощью геометрических фигур

```scss
// float - элемент, который будет обрезан по фигуре
img {
  float: left;
  shape-outside: circle(50%);
}
```

или с помощью shape-outside

```scss
.shape {
  height: 150px;
  width: 150px;
  padding: 20px;
  margin: 20px;
  border-radius: 50%;
  float: left;
  shape-outside: border-box;
}
```

## clearfix

```scss
// на элемент, который является оберткой для элементов float и те, которые должны обтекать
.wrapper::after {
  content: "";
  clear: none | both | right | left; //какое из обтекании отменить
  display: block;
}

// современное решение очистки от обтекания

.wrapper {
  display: flow-root;
}
```

очистить float можно с помощью

```scss
// класс контейнера
.wrapper {
  overflow: auto;
}
```

Недостатки:

- вертикальное выравнивание
- распределение пространства

элемент float изымается из потока и крепится слева или справа, элемент который ниже расположится с противоположной стороны

## BP. Сетка на float

```html
<!-- класс, который определяет ряд -->
<!-- каждый ряд элементов обернут в этот класс -->
<section class="section-features">
  <div class="row">
    <div class="col-1-of-4">...</div>
  </div>
</section>
```

```scss
.row {
  // $grid-width: 114rem === 1440px
  max-width: $grid-width;
  margin: 0 auto;

  //убрать margin с последнего
  &:not(:last-child) {
    margin-bottom: $gutter-vertical;

    @include respond(tab-port) {
      margin-bottom: $gutter-vertical-small;
    }
  }

  @include respond(tab-port) {
    max-width: 50rem;
    padding: 0 3rem;
  }

  @include clearfix;

  //выбрать все элементы, которые начинаются с 'col-'
  [class^="col-"] {
    // обтекание слева
    float: left;

    &:not(:last-child) {
      //Справа margin у всех, кроме последнего
      margin-right: $gutter-horizontal;

      @include respond(tab-port) {
        margin-right: 0;
        margin-bottom: $gutter-vertical-small;
      }
    }

    @include respond(tab-port) {
      width: 100% !important;
    }
  }

  // ширина 1 из 2 колонок: из 100% убираем правый, последний margin и делим на 2
  .col-1-of-2 {
    width: calc((100% - #{$gutter-horizontal}) / 2);
  }
  // ширина 1 из 3 колонок: из 100% убираем 2 правых margin и делим на 3
  .col-1-of-3 {
    width: calc((100% - 2 * #{$gutter-horizontal}) / 3);
  }
  // ширина колонки, которая занимает 2/3: из 100% убираем 2 правых margin и делим на 3
  .col-2-of-3 {
    width: calc(
      2 * ((100% - 2 * #{$gutter-horizontal}) / 3) + #{$gutter-horizontal}
    );
  }

  .col-1-of-4 {
    width: calc((100% - 3 * #{$gutter-horizontal}) / 4);
  }

  .col-2-of-4 {
    width: calc(
      2 * ((100% - 3 * #{$gutter-horizontal}) / 4) + #{$gutter-horizontal}
    );
  }

  .col-3-of-4 {
    width: calc(
      3 * ((100% - 3 * #{$gutter-horizontal}) / 4) + 2 * #{$gutter-horizontal}
    );
  }
}
```

# PB. Первая заглавная буква

```html
<p class="has-dropcap">Carol McKinney Highsmith ...</p>
```

```scss
.has-dropcap:first-letter {
  font-family: "Source Sans Pro", Arial, Helvetica, sans-serif;
  float: left;
  font-size: 6rem;
  line-height: 0.65;
  margin: 0.1em 0.1em 0.2em 0;
}
```
