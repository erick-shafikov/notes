<!-- расположение текста в контейнере -------------------------------------------------------------------------------------------------------->

# расположение текста в контейнере

## text-align

CSS-свойство описывает, как линейное содержимое, наподобие текста, выравнивается в блоке его родительского элемента. text-align не контролирует выравнивание элементов самого блока, но только их линейное содержимое.

```scss
.text-align {
  text-align: left;
  text-align: right;
  text-align: center;
  text-align: justify;
  text-align: start;
  text-align: end;
  text-align: match-parent; //c учетом direction
  text-align: start end;
  text-align: "."; // до символа
  text-align: start ".";
  text-align: "." end;
}
```

так же есть свойство text-align-last позволяющее определить расположение последней перенесенной строки

## alignment-baseline (-ff)

Свойство CSS определяет определенную базовую линию, используемую для выравнивания текста блока и содержимого на уровне строки. Выравнивание базовой линии — это отношение между базовыми линиями нескольких объектов выравнивания в контексте выравнивания

```scss
.alignment-baseline {
  alignment-baseline: alphabetic;
  alignment-baseline: central;
  alignment-baseline: ideographic;
  alignment-baseline: mathematical;
  alignment-baseline: middle;
  alignment-baseline: text-bottom;
  alignment-baseline: text-top;

  /* Mapped values */
  alignment-baseline: text-before-edge; /* text-top */
  alignment-baseline: text-after-edge; /* text-bottom */
}
```

## dominant-baseline

Свойство CSS определяет определенную базовую линию, используемую для выравнивания текста и содержимого на уровне строки в блоке.

```scss
.dominant-baseline {
  dominant-baseline: alphabetic;
  dominant-baseline: central;
  dominant-baseline: hanging;
  dominant-baseline: ideographic;
  dominant-baseline: mathematical;
  dominant-baseline: middle;
  dominant-baseline: text-bottom;
  dominant-baseline: text-top;
}
```

## alignment-baseline (-ff)

Выравнивание текста в сетках и svg

```scss
.alignment-baseline {
  alignment-baseline: baseline;

  /* Keyword values */
  alignment-baseline: alphabetic;
  alignment-baseline: central;
  alignment-baseline: ideographic;
  alignment-baseline: mathematical;
  alignment-baseline: middle;
  alignment-baseline: text-bottom;
  alignment-baseline: text-top;

  /* Mapped values */
  alignment-baseline: text-before-edge; /* text-top */
  alignment-baseline: text-after-edge; /* text-bottom */

  /* Deprecated values  */
  alignment-baseline: auto;
  alignment-baseline: before-edge;
  alignment-baseline: after-edge;
  alignment-baseline: hanging;
}
```
