# @ директивы

## @scope

Позволяет определить стили для поддеревьев

```scss
 @scope (scope root) to (scope limit) {
  rulesets
}
```

# функции

## host-context()

Может использоваться вместе с псевдо классом scope

# псевдо классы

## :defined

работает с пользовательскими элементами объявленные CustomElementRegistry.define(),

## :host

выбирает элемент, к которому прикреплен теневой элемент

## :state()

для определенного состояния компонента

```scss
.labeled-checkbox {
  border: dashed red;
}
.labeled-checkbox:state(checked) {
  border: solid;
}
```

# псевдо элементы

# ::part()

для настройки теневого dom из глобальных стилей

```html
<template id="star-rating-template">
  <form part="formPart">
    <fieldset part="fieldsetPart"></fieldset></form
></template>
```

```scss
star-rating::part(formPart) {
  /* styles */
}
star-rating::part(fieldsetPart) {
  /* styles */
}
```

# ::slotted()

выбирает элемент slot по переданному селектору
