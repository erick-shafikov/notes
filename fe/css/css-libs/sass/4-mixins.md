# Миксины

```scss
// Создание статичного миксина
@mixin clearFix {
  &::after {
    content: "";
    clear: both;
    display: table;
  }
}

// Использование
nav {
  margin: 30px;
  background-color: $color-primary;
 
  @include clearFix;
}

```

## параметры примесей

```scss
@mixin styled-link-text($color: "red", $radius: 10px, $selectors...) {
  //red - параметр по умолчанию
  //selectors - остаточные параметры (*)
  text-decoration: none;
  text-transform: uppercase;
  color: $color;
@if $radius != 0 {
    //проверка на наличие аргумента
    border-radius: $radius;
  }
  //использование остаточных параметров миксина (*)
  @for $i from 0 to length($selectors) {
    #{nth($selectors, $i + 1)} {
      position: absolute;
      height: $height;
      margin-top: $i * $height;
    }
  }
}
```
