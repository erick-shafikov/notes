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
  //  & + p
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
