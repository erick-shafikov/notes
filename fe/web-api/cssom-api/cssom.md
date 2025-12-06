Позволяет манипулировать css из js

события:

- AnimationEvent

Интерфейсы:

- CSS - реализует два стат метода
- - supports() - поддерживается для пара ключ-значение
- - escape() - для вывода строки

- CSSConditionRule - для условий

```js
/* css 
@media (min-width: 500px) {
  body {
    color: blue;
  }
}
*/

const targetRule = document.styleSheets[0].cssRules[0];
console.log(targetRule.conditionText); // "(min-width: 500px)"
```

- CSSCounterStyleRule (родитель - CSSRule) - позволяет работать с счетчиками
- свойства экз:
- - name

```js
/* 
@counter-style box-corner {
  system: fixed;
  symbols: ◰ ◳ ◲ ◱;
  suffix: ": ";
  fallback: disc;
}
*/
let myRules = document.styleSheets[0].cssRules;
console.log(myRules[0].name); // "box-corner"
```

- - system
- - symbols
- - additiveSymbols
- - negative
- - prefix
- - suffix
- - range
- - pad
- - speakAs
- - fallback

- CSSFontFaceRule - (родитель - CSSRule) позволяет работать с font-face
- CSSFontFeatureValuesRule
- CSSImportRule (родитель - CSSRule)
- - CSSImportRule.href
- - CSSImportRule.layerName
- - CSSImportRule.media
- - CSSImportRule.styleSheet
- - CSSImportRule.supportsText

```js
// @import url("style.css") screen;

const myRules = document.styleSheets[0].cssRules;
console.log(myRules[0]); // A CSSImportRule instance object
```

- CSSKeyframeRule
- - CSSKeyframeRule.keyText
- - CSSKeyframeRule.style

```js
/* 
@keyframes slide-in {
  from {
    transform: translateX(0%);
  }

  to {
    transform: translateX(100%);
  }
}

*/

let myRules = document.styleSheets[0].cssRules;
let keyframes = myRules[0]; // a CSSKeyframesRule
console.log(keyframes[0]); // a CSSKeyframeRule representing an individual keyframe.
```

- CSSKeyframesRule
- свойства экз:
- - CSSKeyframesRule.name
- - CSSKeyframesRule.cssRules
- - CSSKeyframesRule.length
- методы экз:
- - CSSKeyframesRule.appendRule()
- - CSSKeyframesRule.deleteRule()
- - CSSKeyframesRule.findRule()
