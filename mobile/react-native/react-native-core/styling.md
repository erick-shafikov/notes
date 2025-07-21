# Выбор платформы

Можно назвать файлы компонентов как Component.android.js и Component.ios.js но в импортировать там, где нужно как Component

# StyleSheet - объект, который создает классы

методы

compose - позволяет объединить два объекта стиля, конфликты переопределяются

```js
const styles1 = StyleSheet.create({
  container: {},
  text: {},
});
const styles2 = StyleSheet.create({
  container: {},
  text: {},
});

const container = StyleSheet.compose(styles1.container, styles2.container);
const text = StyleSheet.compose(styles1.text, styles2.text);
```

## методы

- create - позволяет создать объект стилей
- flatten - объединяет стили в один плоский объект принимает в виде объекта

```js
const flattenStyle = StyleSheet.flatten([page.text, typography.header]);
```

## пол

- absoluteFillObject- - шорткат для позиционирования

```js
absoluteFillObject = {
  position: "absolute",
  top: 0,
  left: 0,
  bottom: 0,
  right: 0,
};
// применение
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  box3: {
    ...StyleSheet.absoluteFillObject,
    top: 0,
    left: 0,
    width: 100,
    height: 100,
    backgroundColor: "green",
  },
});
```

- hairlineWidth - создает сплошную горизонтальную линию

```js
const styles = StyleSheet.create({
  __TESTING__: {
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});
```

# Components style props

## Image

```js
const ImageStyleProps = {
  backfaceVisibility: "visible" | "hidden",
  backgroundColor: Color,
  borderBottomLeftRadius: Number,
  // аналогично borderBottomRightRadius
  borderColor: Color,
  borderRadius: Number,
  //borderTopLeftRadius,  borderTopRightRadius
  borderWidth: Number,
  opacity: 1,
  overflow: "visible" | "hidden",
  objectFit: "cover" | "contain" | "fill" | "scale-down",
  resizeMode: "cover" | "contain" | "stretch" | "repeat" | "center",
  tintColor: Color,
  //Android only props
  overlayColor: Color,
};
```

## Layout

Пропсы для контейнера

```js
const LayoutProps = {
  alignContent:
    "flex-start" |
    "flex-end" |
    "center" |
    "stretch" |
    "space-between" |
    "space-around" |
    "space-evenly",
  alignItems: "flex-start" | "flex-end" | "center" | "stretch" | "baseline",
  alignSelf: "flex-start" | "flex-end" | "center" | "stretch" | "baseline",
  aspectRatio: Number | String,
  borderBottomWidth: Number,
  //аналогично borderEndWidth, borderLeftWidth, borderRightWidth, borderStartWidth, borderTopWidth, borderWidth
  bottom: Number, //расстояние внизу от контейнера
  columnGap: Number,
  direction: "inherit" | "ltr" | "rtl",
  display: "none" | "flex",
  // flexDirection  === column, alignContent  === flex-start, flexShrink === 0
  flex: Number, // 1 - весь контейнер, 0 - размерность по width и height, -1 - по minWidth и minHeight
  flexBasis: Number,
  flexDirection: "row" | "row-reverse" | "column" | "column-reverse",
  flexGrow: Number,
  flexShrink: Number,
  flexWrap: "wrap" | "nowrap" | "wrap-reverse",
  gap: Number,
  height: Number, //может быть в процентах
  justifyContent:
    "flex-start" |
    "flex-end" |
    "center" |
    "space-between" |
    "space-around" |
    "space-evenly",
  left: Number, //расстояние от левого края компонента
  //аналогично height, start, end, top
  margin: Number,
  //аналогично marginBottom, marginEnd, marginHorizontal, marginLeft, marginRight, marginStart, marginTop, marginVertical
  maxHeight: Number,
  maxWidth: Number,
  minHeight: Number,
  minWidth: Number,
  overflow: "visible" | "hidden" | "scroll",
  padding: Number,
  //аналогично paddingBottom, paddingEnd, paddingHorizontal, paddingLeft, paddingRight, paddingStart, paddingTop, paddingVertical
  position: "absolute" | "relative",
  rowGap: Number,
  width: Number, //может быть в процентах
  zIndex: Number,
};
```

## Shadow

```js
const ShadowProps = {
  elevation: Number,
  shadowColor: Color,
  // iOS only props
  shadowOffset: {
    width: Number,
    height: Number,
  },
  shadowOpacity: Number,
  shadowRadius: Number,
};
```

## Text

```js
const textProps = {
    color: Color,
    fontFamily: 'FontName',
    fontSize: number,
    fontWeight: 'normal'|'bold'|'100'|'200'|'300'|'400'|'500'|'600'|'700'|'800'|'900',
    fontVariant: 'small-caps'|'oldstyle-nums'|'lining-nums'|'tabular-nums'|'proportional-nums',
    letterSpacing: number,
    lineHeight: number,
    textAlign: 'auto'|'left'|'right'|'center'|'justify',
    textDecorationLine: 'none'|'underline'|'line-through'|'underline line-through',
    textDecorationStyle : 'solid'|'double'|'dotted'|'dashed',
    textShadowColor: 'color',
    textShadowOffset: {width?: number, height?: Number},
    textShadowRadius: 0,
    textTransform: 'none'|'uppercase'|'lowercase'|'capitalize',
    userSelect: 'auto', 'text', 'none', 'contain', 'all',

    //Android only
    includeFontPadding: true,
    textAlignVertical: 'auto' | 'top' | 'bottom' | 'center',
    verticalAlign: 'auto' | 'top' | 'bottom' | 'middle',

    //iOS only
    textDecorationColor: Color
    textDecorationStyle : 'solid' | 'double' | 'dotted' | 'dashed',
    writingDirection: 'auto' | 'ltr' | 'rtl'

  },
```

## View

Пропсы для элементов

```js
const viewStyleProps = {
  backfaceVisibility: "visible" | "hidden",
  backgroundColor: "color",
  borderBottomColor: "color",
  borderRadius: number, //borderBottomEndRadius, borderBottomLeftRadius, borderBottomRightRadius, borderBottomStartRadius, borderStartEndRadius, borderStartStartRadius, borderEndEndRadius, borderEndStartRadius, borderTopEndRadius,borderTopLeftRadius, borderTopRightRadius, borderTopStartRadius
  borderWidth: number, //borderBottomWidth, borderLeftWidth, borderRightWidth, borderTopWidth
  borderColor: "color", //borderEndColor, borderLeftColor, borderRightColor, borderStartColor, borderTopColor
  borderStyle: "solid" | "dotted" | "dashed",
  opacity: number,
  // определяет как будут обрабатываться касания
  // auto - касания обрабатываются
  // none - касания не обрабатываются
  // box-none - касания на view не обрабатываются, только на дочерних
  // box-only - касания обрабатываются только на view,но не на дочерних элементах
  pointerEvents: "auto" | "box-none" | "box-only" | "none",

  // android only
  elevation: Number,
  //iOS only
  borderCurve: "circular" | "continuous",
  cursor: "auto" | "pointer",
};
```
