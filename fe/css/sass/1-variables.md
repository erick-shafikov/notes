# Переменные

```scss
// Использование переменных
$color-primary: #f9ed69;// yellow color

nav {
  margin: 30px;
  background-color: $color-primary;
}

// Значения по умолчанию и их переопределение
//_some-sass-module.scss
$black: #000 !default;
$border-radius: 0.25rem !default;
$box-shadow: 0 0.5rem 1rem rgba($black, 0.15) !default;
.test {
  border-radius: $border-radius;
  box-shadow: $box-shadow;
  width: 200px;
  height: 200px;
}

//main.scss
@import "some-sass-module.scss";
@use "some-sass-module.scss"

with (//переопределение значений переменных при использование в модуле
  $black: #222,
  $border-radius: 0.1rem
);
// Объявление глобальных переменных в любом месте !global
$variable: second-global-value !global;

//Потоковое изменение переменных:

$dark-theme: true !default;
$primary-color: #f8bbd0 !default;
$accent-color: #6a1b9a !default;
@if $dark-theme {
  $primary-color: darken($primary-color, 60%);
  $accent-color: lighten($accent-color, 60%);
}
.button {
  background-color: $primary-color; //к переменной будет применен darken
  border: 1px solid $accent-color; //к переменной будет применен darken
  border-radius: 3px;
}

```
