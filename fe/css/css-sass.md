npm i node-sass

```json
// скрипт для запуска
{
  "main": "index.js",
  "scripts": {
    "watch:sass": "node-sass sass/main.scss css/style.css -w",
    "devserver": "live-server",
    "start": "npm-run-all --parallel devserver watch:sass",
    "compile": "node-sass sass/main.scss css/style.comp.css",
    "concat": "concat -o css/style.concat.css css/style.comp.css",
    "prefix": "postcss --use autoprefixer -b 'last versions' css/style.concat.css -o css/style.prefix.css",
    "compress": "node-sass css/style.prefix.css css/style.css --output-style compressed",
    "build": "npm-run-all compile concat prefix compress"
  },
  "devDependencies": {
    "autoprefixer": "^7.1.4",
    "concat": "^1.0.3",
    "node-sass": "^7.0.1",
    "npm-run-all": "^4.1.1",
    "postcss-cli": "^4.1.1"
  }
}
```

sass/main.scss – где находится sass файл
css/style.css – куда компилировать css файл
-w – флаг watch, отслеживает изменения

npm live-server – утилита для запуска проекта

запуск параллельно два окна

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

# вложенность

```scss
.navigation {
  // .navigation li {…}
  list-style: none;

  li {
    display: inline-block;
    margin: 30px;
  }
}

.navigation {
  //с псевдо классами
  list-style: none;

  li {
    display: inline-block;
    margin-left: 30px;

    &:first-child {
      margin: 0;
    }
  }
}
```

# интерполяции

```scss
//можно создавать конфигурируемые миксины с гибкими полями
@mixin circle($name, $width, $height, $color, $radius) {
  .#{$name} {
    #{$width}: 100px;
    #{$height}: 100px;
    background-color: #{$color};
    border-radius: #{$radius};
  }
}

@include circle("test", height, width, red, 50%);
```

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
