<!-- Добавление стилей ----------------------------------------------------------------------------------------------------------------------->

# Добавление стилей. Связанные стили

Связанные стили – описание селекторов и их значение располагается в отдельном файле с расширением css, с использованием тега <link>, который помещается в контейнер <head>, варианты подключения:

1. Вариант:

```html
<head>
   
  <meta />
   
  <title>…</title>
    <link rel=stylesheets href=./myStyle.css>
  <!--можно подключить стили, которые лежат на компе-->
   
  <link rel="stylesheets" href="www…" />
  <!-- подключить стили с другого сайта -->
</head>
```

2. Вариант:
   С помощью JS, создаем скрипт с тегом link или style

## Глобальные и внутренние стили

### Глобальный стиль(в head)

```html
<head>
  <meta />
   
  <title></title>
  <style>
    h1{
      background-color: red
    }       
  </style>
</head>
```

### Внутренний стиль

```html
<head>
   
  <meta />
   
  <title></title>
</head>
   
<p style="color:123">…</p>
```

Приоритет (по убыванию) внутренний, глобальный, связанный

- [импорт CSS поддерживает разные условия](./at-rules.md/#import)

```html
<head>
  <meta />
  <title>…</title>
  <style>
    @import /style/main.css screen; /* Стиль для монитора */
    @import /style/smart.css print, handheld; /* Стиль для печати и смартфона */
  </style>
</head>
```

media позволяет указать тип носителя

```html
<link media="print" handheld rel="stylesheet" href="pront.css" />
<link media="screen" rel="stylesheet" href="main.css" />
```

## Подключение

```scss
//Произвольные строки:
{
  font-family: "Times New Roman", serif
  content: привет
}
// CSS-директивы
@font-face {
  font-family: Open Sans;
  src:
    url(),
    url();
}
@media (max-width: 600px) {//при максимальное ширине отображения sidebar скроется
  .sidebar {
    display: none;
  }
}

```

## нормализация

Нормализация – это файл CSS файл, который обеспечивает кроссбраузерность подключение:

```html
<link rel=stylesheet href=css/normalize.css>
```

Сброс – сброс стилей
Подключение нестандартных шрифтов, Google fonts для шрифтов

сброс стилей для форм.

```scss
button,
input,
select,
textarea {
  font-family: inherit;
  font-size: 100%;
  box-sizing: border-box;
  padding: 0;
  margin: 0;
}

textarea {
  overflow: auto;
}
```

<!-- Поведение при печати ---------------------------------------------------------------------------------------------------------------------------->

# Поведение на разных носителях

## page-break-before, page-break-after, page-break-inside

Устанавливает разрывы для печати на странице до или после элемента

```scss
 {
  page-break-before: auto;
  page-break-before: always;
  page-break-before: avoid;
  page-break-before: left;
  page-break-before: right;
  page-break-before: recto;
  page-break-before: verso;
}
```

## break-after (break-before, break-inside)

Применяется для определения разрыва страницы при печати а также для сетки из колонок

break-inside - управление разрывами внутри колонок
break-before, break-inside - до и после

```scss
 {
  break-after: auto; //не будет форсировать разрыв
  break-after: avoid; //избегать любых переносов до/после блока с
  break-after: always;
  break-after: all;

  /* Page break values */
  break-after: avoid-page;
  break-after: page;
  break-after: left;
  break-after: right;
  break-after: recto;
  break-after: verso;

  /* Column break values */
  break-after: avoid-column;
  break-after: column;

  /* Region break values */
  break-after: avoid-region;
  break-after: region;
}
```

## widows

определяет какое количество линий должно быть в начале страницы

- [touch-action позволяет управлять поведением элемента на тач скрине при увеличении]

```scss
 {
  //
}
```
