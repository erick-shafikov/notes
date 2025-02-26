Советы по улучшению работы с изображениями:

- использование правильных форматов
- компрессия изображений
- lazy loading
- адаптивные изображения
- использование cdn

Использования изображения как фон:

# background

Свойство позволяет добавить изображение, градиент на задний фон элемента

это короткая запись для background-clip + background-color + background-image + background-origin + background-position + background-repeat + background-size + background-attachment

```scss
 {
  //два цвета смешаются ровно посередине
  background: linear-gradient(#e66465, #9198e5);
  // множественный градиент
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

Сокращенная запись

```scss
.box {
  background: linear-gradient(
        105deg,
        rgb(255 255 255 / 20%) 39%,
        rgb(51 56 57 / 100%) 96%
      ) center center / 400px 200px no-repeat, url(big-star.png) center
      no-repeat, rebeccapurple;
}
```

Пример с элементом состоящим из нескольких элементов - изображения и градиента

```html
<div class="multi-bg-example"></div>
```

```scss
.multi-bg-example {
  width: 100%;
  height: 400px;
  background-image: url(firefox.png), url(bubbles.png), linear-gradient(to right, rgba(30, 75, 115, 1), rgba(255, 255, 255, 0));
  background-repeat: no-repeat, no-repeat, no-repeat;
  background-position: bottom right, left, right;
}
```

## background-attachment

Определяет поведения заднего фона при прокрутке

```scss
 {
  background-attachment: scroll; //изображение позади будет прокручиваться
  background-attachment: fixed; //изображение позади не будет прокручиваться
  background-attachment: local; //в зависимости от прокручивания контента позади которого будет изображение
}
```

## background-clip

Настраивает как будет обрезаться изображение, которое находится позади

```scss
 {
  background-clip: border-box; //до края границы
  background-clip: padding-box; // до края отступа
  background-clip: content-box; // внутри содержимого
  background-clip: text; //обрезка текстом
}
```

## background-image

Может быть как градиентом так и изображением

```scss
 {
  background-image: linear-gradient(black, white);
  background-image: url("image.png");

  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png); // несколько изображений

  //Создание изображения с наложением градиента
  background-image: linear-gradient(
      rgba($color-secondary, 0.93),
      rgba($color-secondary, 0.93)
    ), url("../img/hero.jpeg");
  //пример с распределением градиента
  background-image: linear-gradient(
      105deg,
      rgba($color-white, 0.9) 0%,
      rgba($color-white, 0.9) 50%,
      rgba($color-white, 0.9),
      transparent 50%
    ), url("../img/nat-10.jpg");
}
```

## background-origin

как расположить изображение относительно рамок и контента

```scss
 {
  background-repeat: no-repeat; // сначала нужно отключить повтор изображения
  //позиционирование
  background-origin: border-box; //растянуть по всему контейнеру, Фон располагается относительно рамки.
  background-origin: padding-box; //не включая рамки, Фон расположен относительно поля отступа.
  background-origin: content-box; //только по границам контента, Фон располагается относительно поля содержимого.
}
```

## background-position

```scss
 {
  // прилепить к краям
  background-position: top;
  background-position: bottom;
  background-position: left;
  background-position: right;
  background-position: center;
  // сдвиг в процентах и единицах
  background-position: 25% 75%;
  background-position: 0 0;
  background-position: 1cm 2cm;
  background-position: 10ch 8em;
  // точное расположение относительно краев
  background-position: bottom 10px right 20px;
  background-position: right 3em bottom 10px;
  background-position: bottom 10px right;
  background-position: top right 10px;
  // для нескольких изображений
  background-position: 0 0, center;

  // несколько изображений
  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png);
  background-repeat: no-repeat, repeat-x, repeat; // для image1.png будет применено no-repeat так как свойства применяются циклично
}
```

### background-position-x и background-position-y

определяет горизонтальную позицию изображения

```scss
 {
  background-position-x: left;
  background-position-x: center;
  background-position-x: right;

  /* <percentage> values */
  background-position-x: 25%;

  /* <length> values */
  background-position-x: 0px;
  background-position-x: 1cm;
  background-position-x: 8em;

  /* Side-relative values */
  background-position-x: right 3px;
  background-position-x: left 25%;

  /* Multiple values */
  background-position-x: 0px, center;
}
```

## background-repeat

```scss
.background-repeat {
  background-repeat: repeat; // по умолчанию повтор включен
  background-repeat: repeat-x; // repeat no-repeat
  background-repeat: repeat-y; // no-repeat repeat
  background-repeat: space; // будет заполнено не обрезая изображения
  background-repeat: round; // по рамке
  background-repeat: no-repeat; // отключить повтор

  // варианты повтора изображения можно задавать для вертикальной и горизонтальной осей
  background-repeat: repeat space;
  background-repeat: repeat repeat;
  background-repeat: round space;
  background-repeat: no-repeat round;
}
```

## background-size

Управление размером изображения

```scss
.background-size {
  // оба свойства не меняют свои пропорции
  background-size: cover; // cover - не будет тянуть изображение, при уменьшении будет обрезаться
  background-size: contain; //  contain - растянет или сузит по всем блоку но изменит не пропорции

  /* Указано одно значение - ширина изображения, */
  /* высота в таком случае устанавливается в auto */
  background-size: 50%;
  background-size: 3em;
  background-size: 12px;
  background-size: auto; // растягивает сохраняя пропорции

  // два значения - по горизонтали и вертикали
  background-size: 50% auto;
  background-size: 3em 25%;
  background-size: auto 6px;
  background-size: auto auto;
  // растянет изображения меняя пропорции
  background-size: 100% 100%;

  /* Значения для нескольких фонов */
  /* Не путайте такую запись с background-size: auto auto */
  background-size: auto, auto;
  background-size: 50%, 25%, 25%;
  background-size: 6px, auto, contain;
}
```

## -------------------------------------------------------

## background-blend-mode

Определяет как будут смешиваться наслаиваемые цвета и изображения

как background-image будет смешиваться с вышележащими слоями

правило смешивания наслаивающих изображений и фонов https://developer.mozilla.org/en-US/docs/Web/CSS/blend-mode

Значений всего 16

```scss
 {
  background-blend-mode: darken, luminosity;
}
```

# object-fit

Позволяет тегу img определить размеры относительно контейнера

```scss
.object-fit {
  object-fit: fill; //заполняет весь контейнер, меняя свои пропорции
  object-fit: contain; //растянет под контейнер, но оставит пропорции
  object-fit: cover; //оставит пропорции, но поместит в контейнер часть изображения
  object-fit: none; //подстроится под изображение
  object-fit: scale-down; //выберет меньший между none и contain
}
```

# object-position

расположит изображение в контейнере

```scss
 {
  object-position: center top;
  object-position: 100px 50px;
}
```

- [тени](./block-model.md#box-shamask-border-modedow)

# image-свойства:

## image-orientation

none | from-image позволяет клиенту автоматически перевернуть изображение

## image-rendering

auto | crisp-edges | pixelated позволяет сгладить края при возникновении пикселей в изображении

## image-resolution (-) управление качеством

# -moz-force-broken-image-icon (-)

отображать или нет у изображений, которые не удалось загрузить иконку картинки, у которых есть alt атрибут

```scss
.moz-force-broken-image-icon {
  moz-force-broken-image-icon: 0; //нет
  moz-force-broken-image-icon: 1; //да
}
```

<!--  image-set() ---------------------------------------------------------------------------------------------------------------------------->

# image-set()

Позволяет выбрать наиболее подходящее изображение

```scss
.box {
  background-image: url("large-balloons.jpg");
  background-image: image-set(
    "large-balloons.avif" type("image/avif"),
    "large-balloons.jpg" type("image/jpeg")
  );
}

.image-set {
  background-image: image-set("image1.jpg" 1x, "image2.jpg" 2x);
  background-image: image-set(url("image1.jpg") 1x, url("image2.jpg") 2x);
  // Select gradient based on resolution
  background-image: image-set(
    linear-gradient(blue, white) 1x,
    linear-gradient(blue, green) 2x
  );
  // Select image based on supported formats
  background-image: image-set(
    url("image1.avif") type("image/avif"),
    url("image2.jpg") type("image/jpeg")
  );
}
```

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs

## BP. Дефолтный стиль для img

!!! При работе с изображениями

```scss
//запретить вытекание за родительский контейнер
 {
  max-width: 100%;
  height: auto;
}
```

## BP. Центрирование изображения

```css
img {
  display: block;
  margin: 0 auto;
}
```
