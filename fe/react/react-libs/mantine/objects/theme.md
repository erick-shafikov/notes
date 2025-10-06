# theme:

Поля объекта:

## activeClassName и focusClassName

- activeClassName - стиль для активных элементов

activeClassName : '' - для отключения фокусировки

- focusClassName - позволяет добавить класс для фокусировки

```scss
/* в случае клика */
.focus {
  &:focus {
    outline: 2px solid var(--mantine-color-red-filled);
    outline-offset: 3px;
  }
}

/* если навигация с клавиатуры */
.focus-auto {
  &:focus-visible {
    outline: 2px solid var(--mantine-color-red-filled);
    outline-offset: 2px;
  }
}
```

## autoContrast и luminanceThreshold

- autoContrast -контролирует нужно ли изменять текст внутри компонентов, и меняет либо на theme.white или theme.black в зависимости но luminanceThreshold
- luminanceThreshold порог для autoContrast

## black & ## white

Цвета связанные с autoContrast:

- black цвет, применится к тексту ниже чем threshold
- white цвет, применится к тексту выше чем threshold

## breakpoints

для добавления breakpoint-ов

## colors

для добавления цветов

```ts
const theme = createTheme({
  colors: {
    // Add your color
    deepBlue: [
      "#eef3ff",
      "#dce4f5",
      "#b9c7e2",
      "#94a8d0",
      "#748dc1",
      "#5f7cb8",
      "#5474b4",
      "#44639f",
      "#39588f",
      "#2d4b81",
    ],
    // or replace default theme color
    blue: [
      "#eef3ff",
      "#dee2f2",
      "#bdc2de",
      "#98a0ca",
      "#7a84ba",
      "#6672b0",
      "#5c68ac",
      "#4c5897",
      "#424e88",
      "#364379",
    ],
  },
});
```

## components

Объект ключи которого принимают названия компонентов, значение:

- defaultProps - дефолтные значения
- classNames - переопределенные названия классов
- styles - функция для формирования стиля

```ts
type TStyle = (theme, props) => ({ selector: { ... } })
```

Варианты добавления

```ts
const theme = {
  components: {
    Button: {
      defaultProps: {},
      classNames: {},
      styles: {},
    },
    // но лучше через extend, для ts
    // так же будут доступны поля из other
    Button: Button.extend({
      defaultProps: { color: "cyan", variant: "outline" },
      classNames: { root: "my-btn-root", label: "my-btn-label" },
      // или
      classNames: {
        root: classes.root,
        input: classes.input,
        label: classes.label,
      },
      styles: (theme, props) => ({
        root: {
          padding: props.size === "md" ? theme.spacing.md : theme.spacing.sm,
        },
      }),
    }),
  },
};
```

взаимодействие с полем other

```ts
const theme = {
  components: {
    Button: Button.extend({
      styles: (theme, props) => ({
        root: {
          padding:
            props.size === "md" ? theme.spacing.md : theme.other.customProp,
        },
      }),
    }),
  },
};
```

## cursorType

'default' - возможные значения: 'default', 'pointer'

## defaultRadius

'xs' | 'sm' | 'md' | 'lg' | 'xl'

## defaultGradient

объект с полями для компонентов у которых есть variant === 'gradient'

```ts
type DefaultGradient = {
  from: string;
  to: string;
  deg: number;
};
```

## focusRing

'auto', показывать ли фокусировку, возможные значения: auto, always, never

## fontFamily

шрифты по умолчанию

## fontFamilyMonospace

для Monospace шрифтов

## fontSizes

добавления вариантов шрифтов

## fontSmoothing

true - применять ли font-smoothing к телу

## headings

```ts
type Headings = {
  fontFamily: string;
  fontWeight: string;
  textWrap: "wrap" | "nowrap" | "balance" | "pretty" | "stable";
  sizes: {
    h1: HeadingStyle;
    h2: HeadingStyle;
    h3: HeadingStyle;
    h4: HeadingStyle;
    h5: HeadingStyle;
    h6: HeadingStyle;
  };
};
```

```ts
import { createTheme } from "@mantine/core";

const theme = createTheme({
  headings: {
    fontFamily: "Roboto, sans-serif",
    sizes: {
      h1: { fontSize: "36px" },
    },
  },
});
```

## lineHeights

Добавление вариантов line-height

## other

Позволяет добавить любые значения в тему, которые будут доступны в теме

## primaryColor

цвет по умолчанию

## primaryShade

{ light: 6, dark: 8 }, индекс из theme.colors[color]

```ts
const theme = {
  primaryShade: 6,
  // или
  primaryShade: { light: 6, dark: 7 },
};
```

## radius

для добавления border-radius компонентам

# respectReducedMotion

false

## scale

1 === 100%/16px - значение по умолчанию;

## shadows

добавление вариантов box-shadow

```ts
import { createTheme } from "@mantine/core";

const theme = createTheme({
  shadows: {
    md: "1px 1px 3px rgba(0, 0, 0, .25)",
    xl: "5px 5px 3px rgba(0, 0, 0, .25)",
  },
});
```

## spacing

добавление вариантов отступа

# variantColorResolver

функция, которая возвращает цвет в зависимости от варианта. (Button, ActionIcon, ThemeIcon)

<!-- --------------------------------------------------- -->

# слияние двух тем

```tsx
import {
  createTheme,
  MantineProvider,
  mergeThemeOverrides,
} from "@mantine/core";

const theme1 = createTheme({
  // ...
});

const theme2 = createTheme({
  // ...
});

const myTheme = mergeThemeOverrides(theme1, theme2);

function Demo() {
  return <MantineProvider theme={myTheme}></MantineProvider>;
}
```

# Использование темы в компонентах

с помощью [useMantineThemeHook](../hooks/useMantineThemeHook.md)

# тема по умолчанию

```ts
import { DEFAULT_THEME } from "@mantine/core";
```
