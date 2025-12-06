API загрузки шрифтов помогает избавиться от задержки загрузки. Так как шрифты извлекаются в тот момент, когда они нужны

```js
const font = new FontFace("my-font", "url(my-font.woff)", {
  style: "italic",
  weight: "400",
  stretch: "condensed",
});

document.fonts.add(font);
//загрузка
font.load();
//если загружен
document.fonts.ready.then(() => {
  // Use the font to render text (for example, in a canvas)
});
```

# FontFace

конструктор

FontFace(family, source, descriptors):

- family == @font-face
- source - URL-адрес файла шрифта, ArrayBuffer или TypedArray
- descriptors:
- - ascentOverride
- - descentOverride
- - display
- - featureSettings
- - lineGapOverride
- - stretch
- - style
- - unicodeRange
- - variationSettings
- - weight

свойства экземпляра:

- display ⇒ CSSOMString
- family ⇒
- featureSettings ⇒
- loaded ⇒
- status ⇒ "unloaded", "loading", "loaded","error"
- stretch ⇒
- style ⇒
- unicodeRange ⇒
- variant ⇒
- weight ⇒

методы:

- load() ⇒ Promise Загружает шрифт

# FontFaceSet

позволяет управлять загрузкой шрифтов

свойства экземпляра:

- status
- ready
- size

события:

- loading
- loadingdone
- loadingerror

методы:

- add()

```js
const font = new FontFace("MyFont", "url(myFont.woff2)");
document.fonts.add(font);
```

- check()
- clear()
- delete()
- forEach()
- has()
- keys()
- load()
- values()

```js
async function isReady() {
  let ready = await document.fonts.ready;
  console.log(ready);
}

isReady();
```

# FontFaceSetLoadEvent
