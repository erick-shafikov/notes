<!-- Курсор ---------------------------------------------------------------------------------------------------------------------------------->

# Курсор

## cursor

определение вида курсора, при наводки на элемент

```scss
// типы стандартных курсоров
cursor: auto;
cursor: pointer;
cursor: zoom-out;
cursor: context-menu;
cursor: help;
cursor: pointer;
cursor: progress;
cursor: wait;
cursor: cell;
cursor: crosshair;
cursor: text;
cursor: vertical-text;
cursor: alias;
cursor: copy;
cursor: move;
cursor: no-drop;
cursor: not-allowed;
cursor: all-scroll;
cursor: col-resize;
cursor: row-resize;
cursor: n-resize;
cursor: e-resize;
cursor: s-resize;
cursor: w-resize;
cursor: ne-resize;
cursor: nw-resize;
cursor: se-resize;
cursor: sw-resize;
cursor: ew-resize;
cursor: ns-resize;
cursor: nesw-resize;
cursor: nwse-resize;
cursor: zoom-in;
cursor: zoom-out;
cursor: grab;
cursor: grabbing;

// использование изображения в качестве курсора + fallback
cursor: url(hand.cur), pointer;

// использование изображения в качестве курсора + координаты + fallback
cursor: url(cursor_1.png) 4 12, auto;
cursor: url(cursor_2.png) 2 2, pointer;

// использование изображения в качестве курсора + координаты + fallback в виде других изображений
cursor: url(cursor_1.svg) 4 5, url(cursor_2.svg), /* …, */ url(cursor_n.cur) 5 5,
  progress;
```

## pointer-events

Определяет цель для курсора
позволяет указать, что может быть целью курсора

```scss
 {
  pointer-events: auto;
  pointer-events: none;
  // для svg
  pointer-events: visiblePainted;
  pointer-events: visibleFill;
  pointer-events: visibleStroke;
  pointer-events: visible;
  pointer-events: painted;
  pointer-events: fill;
  pointer-events: stroke;
  pointer-events: bounding-box;
  pointer-events: all;
}
```

<!-- псевдоклассы состояний мыши и клавиатуры ------------------------------------------------------------------------------------------------>

# псевдоклассы состояний мыши и клавиатуры

## :focus-visible

работает по разном для клавиатуры и мыши при фокусировке

## :focus-within

элементу с фокусом или элементу с потомком, на котором фокус

## :hover

активизируется когда курсор мыши находится в пределах элемента, но щелчка по нему не происходит
