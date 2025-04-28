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

# CSSStyleValue

базовый класс через который работают остальные

## CSSImageValue

интерфейс для изображений

## CSSKeywordValue

для ключевых слов

## CSSMathValue

для числовых значений

## CSSNumericValue

для математических операций

## CSSPositionValue

## CSSTransformValue

## CSSUnitValue

## CSSUnparsedValue
