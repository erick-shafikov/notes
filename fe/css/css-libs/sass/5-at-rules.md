# @правила

## @content

@content предоставляет место для контента внутри миксина

```scss
@mixin hover {
  &:not([disabled]):hover {
    @content;
  }
}

.button {
  border: 1px solid black;
  @include hover {
    border-width: 2px;
  }
}
```

Пример с медиа запросами (web-dev/natrous)

```scss
@mixin respond($breakpoint) {
  @if $breakpoint == phone {
    @media only screen and (max-width: 37.5em) {
      @content;
    } //600px 600/16=37.5
  }
  @if $breakpoint == tab-port {
    @media only screen and (max-width: 56.25em) {
      @content;
    } //900px 900/16 = 56.25
  }
  @if $breakpoint == tab-land {
    @media only screen and (max-width: 75em) {
      @content;
    } //1200px 1200/16=75
  }
  @if $breakpoint == big-desktop {
    @media only screen and (min-width: 112.5em) {
      @content;
    } //1800px 1800/16=112.5
  }
}
// использование

@include respond(tab-port) {
  padding: 2rem;
}
```

## extend

```scss
%heading {
  font-family: $font-display;
  font-weight: 400;
}

используем .heading-1 {
  @extend %heading;
  font-size: 4.5rem;
  color: $color-grey-light-1;
  line-height: 1;
}
```

## @Функции

```scss
// Объявление функции:
@function divide($a, $b){
  @return $a/$b;
}
// использование:
nav {
  margin: divide(60, 2) * 1px;
  background-color: $color-primary;
}

// Опциональные аргументы
@function invert($color, $amount: 100%) {
  $inverse: change-color($color, $hue: hue($color) + 180);
  @return mix($inverse, $color, $amount);
}

$primary-color: #036;

.header {
  background-color: invert($primary-color, 80%);
}

// Спред оператор в аргументах
$widths: 50px, 30px, 100px;
.micro {
  width: min($widths...);
}

```

## @import

Позволяет импортировать

```scss
//импортируемый файл должен начинаться с _
//pages/_home.scss

@import "pages/home";
```

# %Placeholders

```scss
//переиспользуемая часть кода
$primary-color: #333;
%placeholder {
  $size: 100px;
  width: $size;
  height: $size;
  border-radius: $size * 0.5;
  background-color: $primary-color;
}
.test {
  @extend %placeholder;
}
```
