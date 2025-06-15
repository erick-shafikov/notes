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

## widows

определяет какое количество линий должно быть в начале страницы

```scss
.widows {
  widows: 2;
  widows: 3;
}
```

## touch-action

позволяет управлять поведением элемента на тач скрине при увеличении

```scss
.touch-action {
  touch-action: auto;
  touch-action: none;
  touch-action: pan-x;
  touch-action: pan-left;
  touch-action: pan-right;
  touch-action: pan-y;
  touch-action: pan-up;
  touch-action: pan-down;
  touch-action: pinch-zoom;
  touch-action: manipulation;
}
```

## @page

Позволяет определить стиль при печати страницы, изменить можно только margin, orphans, widows, и разрывы страницы документа.

```scss
@page {
  margin: 1cm;
}

@page :first {
  margin: 2cm;
}
```

Управление может происходить с помощью псевдоклассов :blank, :first, :left, :right

<!-- печать документов ----------------------------------------------------------------------------------------------------------------------->

# псевдоклассы

## :first

представляя первую страницу документа при печати, [используется с @-правилом @page](../styles-managment.md#page)

```scss
@page :first {
  margin-left: 50%;
  margin-top: 50%;
}
```

## :left

используется с @-правилом @page, предоставляет все левые страницы печатного документа,

## :right

используется с @-правилом @page,
