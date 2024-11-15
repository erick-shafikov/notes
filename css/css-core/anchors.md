<!-- Якоря ------------------------------------------------------------------------------------------------------------------------------->

# Якоря

Позволяют разместить один элемент относительно другого.

```html
<!-- якорь -->
<div class="anchor">⚓︎</div>

<!-- контент якоря -->
<div class="infoBox">
  <p>You can edit this text.</p>
</div>
```

```scss
.anchor {
  anchor-name: --myAnchor;
}

.infoBox {
  position-anchor: --myAnchor;
  // обязательно нужно что бы position === fixed или position === absolute
  position: fixed;
  opacity: 0.8;
  inset-area: top left;
}
```

- [anchor-name для задания имени для компонента-якоря](./css-props.md#anchor-name-якоря)
- [inset-area для расположения контента якоря](./css-props.md#inset-area-якоря)
- [anchor-position для указания имени контента](./css-props.md#position-anchor-якоря)
- [position-area размещение якоря]
- position-try-order - для позиционирования
- position-try-order - дял порядке
- position-try = position-try-order + position-try-order
- position-visibility - отвечает за отражение
- [@правило](./at-rules.md#position-try-якоря)
- [функция для определения позиции якоря](./functions.md#anchor-якоря)
- [измерение для якоря](./functions.md#anchor-size-якоря)
- text-anchor выравнивает блок, содержащий строку текста, где область переноса определяется из свойства

Пример со слайдером, в котором будет отображаться значение шкалы

```html
<label for="slider">Change the value:</label>
<input type="range" min="0" max="100" value="25" id="slider" />
<output>25</output>
<script>
  const input = document.querySelector("input");
  const output = document.querySelector("output");

  input.addEventListener("input", (event) => {
    output.innerText = `${input.value}`;
  });
</script>
```

```scss
input::-webkit-slider-thumb {
  // цепляемся за слайдер
  anchor-name: --thumb;
}

output {
  // цепляемся за слайдер
  position-anchor: --thumb;
  position: absolute;
  left: anchor(right);
  bottom: anchor(top);
}
```
