# оптимизационные intrinsic значения (!!!TODO)

оптимизационные значения contain-intrinsic-block-size, contain-intrinsic-height, contain-intrinsic-inline-size, contain-intrinsic-size, contain-intrinsic-width

# contain-intrinsic-block-size | block-height | inline-size | intrinsic-size | intrinsic-width

Настройка размеров блочных и строчных элементов при ограничении

contain-intrinsic-size = contain-intrinsic-width + contain-intrinsic-height

```scss
.contain-intrinsic {
  contain-intrinsic-block-size: 1000px;
  contain-intrinsic-block-size: 10rem;
  contain-intrinsic-height: 1000px;
  contain-intrinsic-height: 10rem;
  contain-intrinsic-inline-size: 1000px;
  contain-intrinsic-inline-size: 10rem;

  /* auto <length> */
  contain-intrinsic-block-size: auto 300px;
  contain-intrinsic-height: auto 300px;
  contain-intrinsic-inline-size: auto 300px;
}
```
