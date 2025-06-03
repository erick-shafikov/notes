Позволяет стилизовать элементы dom с помощью background-image, border-image, mask-image, нужно

- определить рисунок с помощью registerPaint()
- зарегистрировать
- paint()

```js
registerPaint(
  //имя
  "headerHighlight",
  //класс который рисует
  class {
    //доступ к пользовательским свойствам
    static get inputProperties() {
      return ["--boxColor", "--widthSubtractor"];
    }
    static get inputArguments() {
      return ["<color>"];
    }
    static get contextOptions() {
      return { alpha: true };
    }
    //canvas-like
    //size - размеры бокса
    //props - позволит обратиться к пользовательским свойствам
    paint(ctx, size, props) {
      ctx.fillStyle = "hsl(55 90% 60% / 100%)";
      ctx.fillRect(0, 15, 200, 20); /* order: x, y, w, h */
      //
      size.width * 0.4 - props.get("--widthSubtractor"),
    }
  }
);
//регистрация
CSS.paintWorklet.addModule("nameOfPaintWorkletFile.js");
```

используем в css

```css
.fancy {
  background-image: paint(headerHighlight);
}
```

# PaintWorkletGlobalScope

Интерфейс PaintWorkletGlobalScopeAPI CSS Painting представляет собой глобальный объект, доступный внутри объекта paint Worklet.

# PaintRenderingContext2D

это контекст рендеринга API для рисования в растровое изображение. Он реализует подмножество CanvasRenderingContext2DAPI со следующими исключениями:
