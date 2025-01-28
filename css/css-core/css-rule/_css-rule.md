```scss
//селектор: hover//псевдокласс
a {
  //свойство
  background-color: #f8f8f8 //значение
;
}
```

Типы значений: integer, number, dimension, percentage, color

Единицы измерения:

- ch - глифа 0 шрифта элемента
- em - Размер шрифта родительского элемента.
- ex - x высота шрифта
- lh - высота строки элемента
- rem (root em) - относительно корня, html, у которого задан font-size
- px
- vh - 1% от высоты
- vw - 1% от ширины
- vmin/vmax - 1% от меньшего/большего ширины окна
- процент
- - !!! margin и padding могут быть в процентах. Рассчитывается на основе Inline Блока
- числа (от 0 до 1)

Цвета: 16х, RGB, RGBA, hsl, hsla

```scss
// пример с rgba
.one {
  background-color: rgb(2 121 139, 0.3);
}
```

- {описание стиля}

<!-- Вложенность ----------------------------------------------------------------------------------------------------------------------------->

# Вложенность

Позволяет описывать правила внутри других правил. Разница с препроцессорами - не компилируется, а считывается браузером. Специфичность === :is()

С пробелами

```scss
.parent-rule {
  .child-rule {
  }
}

// равнозначные записи
.parent-rule {
}

.parent-rule .child-rule {
}
```

без пробелов

```scss
.parent-rule {
  &:hover {
  }
}

.parent-rule {
}

.parent-rule:hover {
}
```

С псевдо классами если не добавить амперсанд

```scss
.parent-rule {
  :hover {
  }
}

.parent-rule {
}

.parent-rule *:hover {
}
```

Использование &:

- При объединении селекторов, например, с помощью составных селекторов или псевдоклассов .
- Для обратной совместимости.
- В качестве визуального индикатора

!!!НЕ предусмотрена конкатенация

```scss
.card {
  .featured & {
  }
}
// равнозначные записи
.card {
}

.featured .card {
}
```

```scss
.card {
  .featured & & & {
  }
}

.card {
}

.featured .card .card .card {
}
```

Если использовать амперсанд наверху - то будет относится к внешнему контексту

Вложенности также подчиняются и @-правила

Поддерживает комбинаторы

```scss
h2 {
  color: tomato;
  + p {
    color: white;
    background-color: black;
  }
}

h2 {
  color: tomato;
  & + p {
    color: white;
    background-color: black;
  }
}
.a {
  /* styles for element with class="a" */
  .b {
    /* styles for element with class="b" which is a descendant of class="a" */
  }
  &.b {
    /* styles for element with class="a b" */
  }
}

.foo {
  /* .foo styles */
  .bar & {
    /* .bar .foo styles */
  }
}
```

Можно вкладывать и медиа выражения
