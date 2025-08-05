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
