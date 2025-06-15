# добавление шрифтов на сайт:

```html
<link
  href="http://fonts.googleapis.com/css?family=Open+Sans"
  rel="stylesheet"
  type="text/css"
/>
```

## @font-face

установка шрифтов с помощью

```scss
 html{
  font-family: 'Name of font'; //даем название шрифту
  src: url(); //указываем место, где находится шрифт
  src: local(); //указываем место, где находится шрифт на устройстве пользователя
}

@font-face {
  // Указывает имя шрифта, которое будет использоваться для задания свойств шрифта.
  font-family: "Open Sans";
  src: url("/fonts/OpenSans-Regular-webfont.woff2") format("woff2"), url("/fonts/OpenSans-Regular-webfont.woff") format("woff");
  // использование локальных шрифтов
  src: local("Helvetica Neue Bold"), local("HelveticaNeue-Bold"),
  // ------------------------------------------------------------------
  // Определяет как отображается шрифт, основываясь на том, был ли он загружен и готов ли к использованию.
  font-display: auto;
  font-display: block;
  font-display: swap;
  font-display: fallback;
  font-display: optional;
  // Значение font-stretch
  font-stretch: normal; //  ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded;
  // ------------------------------------------------------------------
  font-style: normal;
  font-style: italic;
  font-style: oblique;
  font-style: oblique 30deg;
  font-style: oblique 30deg 50deg;
  // ------------------------------------------------------------------
  font-weight: normal;
  font-weight: bold;
  font-weight: 400;
  /* Multiple Values */
  font-weight: normal bold;
  font-weight: 300 500;
  //настройки
  font-variant: ;
  font-feature-settings: ;
  font-variation-settings: ;
}
```

использование

```scss
html {
  font-family: "myFont", "Bitstream Vera Serif", serif;
}
```

## font-face (js)

[возможность управлять шрифтами через js](../../js/web-api/font-face.md)

<!-- Производительность ---------------------------------------------------------------------------------------------------------------------->

# Производительность

## text-rendering

```scss
.text-rendering {
  text-rendering: auto;
  text-rendering: optimizeSpeed;
  text-rendering: optimizeLegibility;
  text-rendering: geometricPrecision;
}
```
