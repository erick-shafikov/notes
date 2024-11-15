Интерфейс FontFace представляет единый используемый шрифт. Он позволяет контролировать источник шрифта, являющийся URL-адресом внешнего ресурса или буфера; а также контролировать момент загрузки шрифта и его текущее состояние.

```ts
abstract class FontFace {
  name: string;
  ascentOverride: string;
descentOverride: string;
// display: string;
// family: string;
// featureSettings: string;
// lineGapOverride: string;
// loaded: string;
// status: string;
// stretch: string;
// style: string;
// unicodeRange: string;
// variantNon-standard: string;
// variationSettingsExperimental: string;
// weight: string;
// Instance methods: string;

  constructor(family: string, source: string, descriptors: Descriptors) {
    // family - имя
    // source - url
  };

   abstract display(): void;

  abstract load(): Promise;
}


type Descriptors {

}
```

```js
const canvas = document.getElementById("js-canvas");

// load the "Bitter" font from Google Fonts
const fontFile = new FontFace(
  "FontFamily Style Bitter",
  "url(https://fonts.gstatic.com/s/bitter/v7/HEpP8tJXlWaYHimsnXgfCOvvDin1pK8aKteLpeZ5c0A.woff2)"
);
document.fonts.add(fontFile);

fontFile.load().then(
  () => {
    // font loaded successfully!
    canvas.width = 650;
    canvas.height = 100;
    const ctx = canvas.getContext("2d");

    ctx.font = '36px "FontFamily Style Bitter"';
    ctx.fillText("Bitter font loaded", 20, 50);
  },
  (err) => {
    console.error(err);
  }
);
```
