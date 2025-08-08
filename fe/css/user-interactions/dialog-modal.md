## ::backdrop

это прямоугольник с размерами окна, который отрисовывается сразу же после отрисовки любого элемента в полноэкранном режиме,
работает в паре с fullscreenApi и dialog.

## ::details-content (-ff, -safari)

представляет контент дял details

```html
<details>
  <summary>Click me</summary>
  <p>Here is some content</p>
</details>
```

```scss
details::details-content {
  background-color: #a29bfe;
}
```
