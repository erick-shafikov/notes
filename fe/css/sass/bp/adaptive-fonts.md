```scss
@mixin adaptiv-font-engine(
  $maxSize,
  $minSize,
  $lineHeightDelta,
  $maxWidth,
  $minWidth
) {
  $fontCoef: $maxSize - $minSize;
  $widthCoef: $maxWidth - $minWidth;
  $size: calc(
    #{$minSize}px + #{$fontCoef} * ((100vw - #{$minWidth}px) / #{$widthCoef})
  );

  font-size: $size;
  line-height: calc(#{$size} + #{$lineHeightDelta}px);
}

@mixin adaptiv-font($desktopFont, $laptopFont, $tabletFont, $mobileFont) {
  @include adaptiv-font-engine(
    // list.nth($desktopFont, 1) вытаскивает из desktopFont элемент по 1 индексу
    list.nth($desktopFont, 1),
    list.nth($laptopFont, 1),
    list.nth($desktopFont, 2) - list.nth($desktopFont, 1),
    $desktop-size-max,
    $desktop-size-min
  );

  @include laptop-media() {
    @include adaptiv-font-engine(
      list.nth($laptopFont, 1),
      list.nth($tabletFont, 1),
      list.nth($laptopFont, 2) - list.nth($laptopFont, 1),
      $laptop-size-max,
      $laptop-size-min
    );
  }

  @include tablet-media() {
    @include adaptiv-font-engine(
      list.nth($tabletFont, 1),
      list.nth($mobileFont, 1),
      list.nth($tabletFont, 2) - list.nth($tabletFont, 1),
      $tablet-size-max,
      $tablet-size-min
    );
  }

  @include mobile-media() {
    @include adaptiv-font-engine(
      list.nth($mobileFont, 1),
      list.nth($mobileFont, 1),
      list.nth($mobileFont, 2) - list.nth($mobileFont, 1),
      $mobile-size-max,
      $mobile-size-min
    );
  }
}

/*
Mixin can get 1, 2, 3 and 4 arguments of tuple of font-size and line-height
1 - all
2 - other, mobile
3 - other, tablet, mobile
4 - desctop, laptop, tablet, mobile
*/
@mixin font-size($args...) {
  @if (list.length($args) == 1) {
    @include adaptiv-font(
      list.nth($args, 1),
      list.nth($args, 1),
      list.nth($args, 1),
      list.nth($args, 1)
    );
  } @else if (list.length($args) == 2) {
    @include adaptiv-font(
      list.nth($args, 1),
      list.nth($args, 1),
      list.nth($args, 1),
      list.nth($args, 2)
    );
  } @else if (list.length($args) == 3) {
    @include adaptiv-font(
      list.nth($args, 1),
      list.nth($args, 1),
      list.nth($args, 2),
      list.nth($args, 3)
    );
  } @else if (list.length($args) == 4) {
    @include adaptiv-font(
      list.nth($args, 1),
      list.nth($args, 2),
      list.nth($args, 3),
      list.nth($args, 4)
    );
  }
}
```

использование

```scss
@mixin M-Size() {
  @include font-size((20, 26), (18, 22), (16, 20));
}

@mixin M-Medium() {
  font-family: "Roboto-Medium";
  @include M-Size();
}

@mixin M-Regular() {
  font-family: "Roboto-Regular";
  @include M-Size();
}
```
