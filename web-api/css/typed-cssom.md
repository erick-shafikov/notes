# API CSS Typed Object Mode

Позволяет управлять css значениями как объектами JS

CSS Typed OM делает манипуляцию CSS более логичной и производительной

```html
<p>
  <a href="https://example.com">Link</a>
</p>
<dl id="regurgitation"></dl>
```

```js
// Get the element
const myElement = document.querySelector("a");
const stylesList = document.querySelector("#regurgitation");

//с помощью computedStyleMap получим объектную модель
const defaultComputedStyles = myElement.computedStyleMap();

const ofInterest = ["font-weight", "border-left-color", "color", "--color"];

// Iterate through the map of all the properties and values, adding a <dt> and <dd> for each
for (const [prop, val] of defaultComputedStyles) {
  // properties

  //
  stylesList.appendChild(cssProperty);
}
```

# CSSStyleValue

базовый класс

статические методы:

- parse() ⇒ из строки в объект
- parseAll() ⇒ массив объектов

# StylePropertyMap

альтернатива CSSStyleDeclaration

статические методы:

- set() - изменять свойства
- append() - добавит новый css
- delete()
- clear()

# CSSUnparsedValue

для пользовательских свойств

конструктор:

CSSUnparsedValue()

статические методы:

- entries()
- forEach()
- keys()

# CSSKeywordValue

для ключевых слов

конструктор:

CSSKeywordValue()

методы:

- value()

```js
let myElement = document.getElementById("myElement").attributeStyleMap;
myElement.set("display", new CSSKeywordValue("initial"));

console.log(myElement.get("display").value); // 'initial'
```

# CSSStyleValue

базовый класс через который работают остальные

- parse() - вернет объект вида

```js
//CSSTransformValue {0: CSSTranslate, 1: CSSScale, length: 2, is2D: false}
const css = CSSStyleValue.parse(
  "transform",
  "translate3d(10px,10px,0) scale(0.5)"
);
```

- parseAll() ⇒ массив CSSStyleValue

## CSSImageValue

интерфейс для изображений

```js
//<button>Magic Wand</button>
// get the element
/* css
button {
  display: inline-block;
  min-height: 100px;
  min-width: 100px;
  // можем распарить url изображения
  background: no-repeat 5% center url(magic-wand.png) aqua;
}
*/
const button = document.querySelector("button");

// Retrieve all computed styles with computedStyleMap()
const allComputedStyles = button.computedStyleMap();

// Return the CSSImageValue Example
console.log(allComputedStyles.get("background-image"));
console.log(allComputedStyles.get("background-image").toString());
```

## CSSKeywordValue

для ключевых слов, свойства

- value

## CSSMathValue

для числовых значений, статические методы:

- parse

методы экземпляра:

- add
- sub
- mul
- div
- min
- max
- equals
- to - перевод в другие ед измерения
- toSum
- type ⇒ одно из angle, flex, frequency, length, resolution, percent, percentHint, or time

```js
let mathSum = CSS.px("23")
  .add(CSS.percent("4"))
  .add(CSS.cm("3"))
  .add(CSS.in("9"));
// Prints "calc(23px + 4% + 3cm + 9in)"
console.log(mathSum.toString());
```

## CSSNumericValue

для математических операций

## CSSPositionValue

## CSSTransformValue

## CSSUnitValue

## CSSUnparsedValue
