# MUI

1. Material UI — это библиотека компонентов React с открытым исходным кодом, реализующая Material Design от Google. Он комплексный и может быть использован в производстве прямо из коробки.
2. Joy UI — это библиотека компонентов React с открытым исходным кодом, которая реализует собственные принципы проектирования MUI. Он комплексный и может быть использован в производстве прямо из коробки. Компонентов меньше, чем в MUI, но они более гибкие в кастомизации
3. Base UI — это библиотека безголовых («нестилизованных») компонентов React и низкоуровневых хуков. Вы получаете полный контроль над CSS вашего приложения и функциями специальных возможностей.
4. MUI System — это набор утилит CSS, которые помогут вам более эффективно создавать собственные проекты. Это позволяет быстро создавать нестандартные конструкции.

- MUI X
  это набор расширенных компонентов пользовательского интерфейса React для сложных случаев использования. Используйте встроенную интеграцию с пользовательским интерфейсом Material или расширьте свою систему дизайна. В основном предназначена для отображения информации структурированной сложным образом – Чарты и таблицы, компоненты календарей

- Material UI – состоит из библиотеки компонентов, каждый из которых имеет свое Api для работы с этими компонентами. Компоненты делятся на групп по функционалу: Input, Data display, Layout, Utils. Отдельно стоит отметить возможно изменять компоненты с помощью темы, с помощью которой можно добавлять новые значения паллеты, отступов, анимации. Также в теме можно изменять стиль встроенных компонентов.

## кастомизация

Кастомизация компонентов:

- Одноразовая кастомизация, точечная – можно производить с помощью sx пропса, в нем можно указывать измененные стили для вложенных компонентов
- Переисользуемые компоненты – с помощью styled можно изменять компоненты
- Глобальные темы – специальный компонент

## theming

Переменные темы: .palette, .typography, .spacing, .breakpoints, .zIndex, .transitions, .components
!!!Нельзя использовать .vars
Что бы создать тему нужно воспользоваться creteTheme({…Объект с переопределенной темой})

```js
const theme = createTheme({
  status: {
    danger: orange[500],
  },
});
```

Типизация

```ts
declare module "@mui/material/styles" {
  interface Theme {
    status: {
      danger: string;
    };
  } // расширяем ThemeOptions
  interface ThemeOptions {
    status?: {
      danger?: string;
    };
  }
}
```

#### dark mode

Запустить проект по умолчанию в режиме темной темы

```js
const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});
```

Позволяет узнать тему системы, позволяет узнать не только значения palette, но и другие

#### вложенные темы

```tsx
import { ThemeProvider, THEME_ID, createTheme } from "@mui/material/styles";
import { AnotherThemeProvider } from "another-ui-library";
const materialTheme = createTheme(/* your theme */);
function App() {
  return (
    <AnotherThemeProvider>
      <ThemeProvider theme={{ [THEME_ID]: materialTheme }}></ThemeProvider>
    </AnotherThemeProvider>
  );
}
```

#### компоненты с учетом темой

1. Создать слоты для компонента root, value, unit (для примера из доки)
2. Создать композицию из вложенных элементов, добавить поддержку ref-ов
3. Стилизовать компонент через ownerState
4. Добавить поддержку темы помощью хука useThemeProps

```js
const prefersDarkMode = useMediaQuery("(prefers-color-scheme: dark)");
```

Чтобы соединить две темы, можно воспользоваться утилитами

```tsx
import { deepmerge } from "@mui/utils";
import { createTheme } from "@mui/material/styles";
const theme = createTheme(deepmerge(options1, options2));
```

### palette

Цвета: primary, secondary, error, warning, info, success
Токены: main, light, dark, contrastText

Contrast threshold – контраст между фоном и текстом
Добавление нового токена

```js
const theme = createTheme({
  palette: {
    ochre: {
      main: "#E3D026",
      light: "#E9DB5D",
      dark: "#A29415",
      contrastText: "#242105",
    },
  },
});

// Утилита augmentColor
const theme = createTheme(theme, {
  // Custom colors created with augmentColor go here
  palette: {
    salmon: theme.palette.augmentColor({
      color: {
        main: "#FF5733",
      },
      name: "salmon",
    }),
  },
});
```

Типизация

```ts
declare module "@mui/material/styles" {
  interface Palette {
    custom: Palette["primary"];
  }
  interface PaletteOptions {
    custom?: PaletteOptions["primary"];
  }
}

// Для того что бы использовать в компонентах
declare module "@mui/material/Button" {
  interface ButtonPropsColorOverrides {
    custom: true;
  }
}
```

### typography

Можно подключить шрифты удалено и локально
MUI использует rem, 1rem = 14px

Адаптивные шрифты

```js
theme.typography.h3 = {
  fontSize: "1.2rem",
  "@media (min-width:600px)": {
    fontSize: "1.5rem",
  },
  [theme.breakpoints.up("md")]: {
    fontSize: "2.4rem",
  },
};
// или
import { createTheme, responsiveFontSizes } from "@mui/material/styles";
let theme = createTheme();
theme = responsiveFontSizes(theme);
```

Переопределение базовых размеров

```js
const theme = createTheme({
  typography: {
    subtitle1: {
      fontSize: 12,
    },
    body1: {
      fontWeight: 500,
    },
    button: {
      fontStyle: "italic",
    },
  },
});

const theme = createTheme({
  typography: {
    poster: {
      // новый токен типографии
      fontSize: 64,
      color: "red",
    },
    h3: undefined, //Отключение каких-либо вариантов
  },
  components: {
    MuiTypography: {
      defaultProps: {
        variantMapping: {
          poster: "h1", // теперь вариант poster будет для h1
        },
      },
    },
  },
});
```

```ts
declare module "@mui/material/styles" {
  interface TypographyVariants {
    poster: React.CSSProperties; // новый токен типографии
  } // allow configuration using `createTheme`
  interface TypographyVariantsOptions {
    poster?: React.CSSProperties;
  }
} // Update the Typography's variant prop options
declare module "@mui/material/Typography" {
  interface TypographyPropsVariantOverrides {
    poster: true;
    h3: false;
  }
}
```

### spacing

```js
//значения по умолчанию
const theme = createTheme();
theme.spacing(2); // `${8 * 2}px` = '16px'

// Кастомизированные
//через тему
const theme = createTheme({
  spacing: 4,
});
theme.spacing(2); // `${4 * 2}px` = '8px‘
//Bootsptrap - стратегия
const theme = createTheme({
  spacing: (factor) => `${0.25 * factor}rem`,
});
theme.spacing(2); // = 0.25 * 2rem = 0.5rem = 8px

const theme = createTheme({
  spacing: [0, 4, 8, 16, 32, 64],
});
theme.spacing(2); // = '8px'
```

Использование

```js
// -
padding: `${theme.spacing(1)} ${theme.spacing(2)}`, // '8px 16px'
// +
padding: theme.spacing(1, 2), // '8px 16px‘
```

## breakpoints

xs, extra-small: 0px
sm, small: 600px
md, medium: 900px
lg, large: 1200px
xl, extra-large: 1536px

Пример стилизованного компонента с учетом размеров

```js
const Root = styled("div")(({ theme }) => ({
  [theme.breakpoints.down("md")]: {
    //вариант для [0, md) или [0, 900px)
    backgroundColor: "red",
  },
  [theme.breakpoints.up("lg")]: {
    //вариант для  [md, ∞) [900px, ∞)
    backgroundColor: green[500],
  },
  [theme.breakpoints.only("md")]: {
    //вариант для  [md, md + 1) [md, lg) [900px, 1200px)
    backgroundColor: "red",
  },
  [theme.breakpoints.not("md")]: {
    //вариант для  [xs, md) and [md + 1, ∞), [xs, md) and [lg, ∞), [0px, 900px) and [1200px, ∞)
    backgroundColor: "red",
  },
  [theme.breakpoints.between("sm", "md")]: {
    backgroundColor: "red",
  },
}));
```

Переопределение стандартных значений и изменение

```js
const theme = createTheme({
  breakpoints: {
    values: {
      mobile: 0,
      tablet: 640,
      laptop: 1024,
      desktop: 1200,
    },
  },
});
```

Типизация

```ts
declare module "@mui/material/styles" {
  interface BreakpointOverrides {
    xs: false; // removes the `xs` breakpoint
    sm: false;
    md: false;
    lg: false;
    xl: false;
    mobile: true; // adds the `mobile` breakpoint
    tablet: true;
    laptop: true;
    desktop: true;
  }
}
```

### density

Компоненты в который плотность идет пропсом

Button, Fab, FilledInput, FormControl, FormHelperText, IconButton, InputBase, InputLabel, ListItem, OutlinedInput, Table, TextField, Toolbar

### transition

theme.transitions.create(props, options) => transition

```js
const StyledAvatar = styled(Avatar)`
  ${({ theme }) => `
  …
    transition: ${theme.transitions.create(["background-color", "transform"], {
  //анимируем background-color и transform
  duration: theme.transitions.duration.standard, //второй параметр - объект duration, easing, delay
})};
  &:hover {
    background-color: ${theme.palette.secondary.main};
    transform: scale(1.3);
  }
  `}
`;
```

Значения duration по умолчанию

```js
const theme = createTheme({
  transitions: {
    duration: {
      shortest: 150,
      shorter: 200,
      short: 250, // most basic recommended timing
      standard: 300, // this is to be used in complex animations
      complex: 375, // recommended when something is entering screen
      enteringScreen: 225, // recommended when something is leaving screen
      leavingScreen: 195,
    },
  },
});
```

Значения easing по умолчанию

```js
const theme = createTheme({
  transitions: {
    easing: {
      // This is the most common easing curve.
      easeInOut: "cubic-bezier(0.4, 0, 0.2, 1)", // Objects enter the screen at full velocity from off-screen and // slowly decelerate to a resting point.
      easeOut: "cubic-bezier(0.0, 0, 0.2, 1)", // Objects leave the screen at full velocity. They do not decelerate when off-screen.
      easeIn: "cubic-bezier(0.4, 0, 1, 1)", // The sharp curve is used by objects that may return to the screen at any time.
      sharp: "cubic-bezier(0.4, 0, 0.6, 1)",
    },
  },
});
```

## components

в createTheme можно изменить:

1. значение по умолчанию для всех компонентов
   - переписать стили темы
   - сделать состояние основанное на пропсах
   - сделать состояние основанное на стандартных атрибутах из апи компонентов
2. sx – проп
3. Создание новых вариантов для компонента
4. использование переменных темы

## кастомизация компонентов

1. Одноразовая кастомизация с помощью sx пропсы. Переопределение стилей вложенных элементов в стандартные компоненты строится на том, что каждый класс строится по принципу [hash]-Mui[Component name]-[name of the slot].
2. classNames – атрибут. Для каждого псевдо класса у MUI есть свой класс для более высокой специфики active - .Mui-active, checked - .Mui-checked, completed - .Mui-completed, disabled - .Mui-disabled, error - .Mui-error, expanded - .Mui-expanded, focus visible - .Mui-focusVisible, focused - .Mui-focused, readOnly - .Mui-readOnly, required - .Mui-required, selected - .Mui-selected
3. Переисользуемые компоненты
   - без пропсов
   - с пропсами, которые позволяют динамически изменять компонент
   - с переопределяемым объектом стилей, в котором переопределяются значение переменных CSS
4. Переписывание глобальных стилей с помощью темы
5. Переопределение с помощью компонента `<GlobalStyles styles={…} />`. BP для глобальных стилей является определение компонента в переменную

```tsx
const inputGlobalStyles = <GlobalStyles styles={} />;
function Input(props) {
  return (
    <React.Fragment>
      {inputGlobalStyles}
      <input {...props} />     
    </React.Fragment>
  );
}
```

## api design approach

Композиция

- children
- пропы-children-ы
- spread – пропсы передаются от родителя к потомку до root, нежелательно использовать classNames
- классы всегда применяются к root элементы
- стили по умолчанию сгруппированы в один класс
- все стили не применяемые к root имеют префикс
- булевы значения идут без префикса
  CSS – классы
- класс применённый к root элементу называется root
- классы по умолчанию формируют один класс
- Остальные классы имеют префикс
- булевы не имеют префикса
- enum-свойства имеют префикс

Вложенные компоненты имеют

- свой пропы (id для Input)
- xxxProp
- xxxComponent
- xxxRef

Наименование
Если два значения – boolean, если больше, то Enum

## совместимость

CSS
через импорт .css фалов, в которых указаны классы компонентов (для одноразовых компонентов)
Глобальный CSS
Нужно пользоваться встроенным классами .MuiSlider-root, .MuiSlider-root:hover и другими
CSS-модули (аналогично CSS)
Styled
Тема
Emotion - через css-проп

## композиция

При оборачивании компонентов

```jsx
const WrappedIcon = (props) => <Icon {...props} />;
WrappedIcon.muiName = Icon.muiName;
```

component-проп
Обеспечивает полиморфные компоненты, если в качестве компонента указать inline кастомный компонент, то это повлечет за собой потерю состояния, деструктуризация компонента при каждом рендере. Выход обернуть в useMemo

Если передаем react-комопнент, а не обычный html – элемент, то нужно его обернуть в memo

```tsx
import { Link, LinkProps } from "react-router-dom";
function ListItemLink(props) {
  const { icon, primary, to } = props;
  const CustomLink = React.useMemo(
    () =>
      React.forwardRef<HTMLAnchorElement, Omit<RouterLinkProps, "to">>(
        function Link(linkProps, ref) {
          return <Link ref={ref} to={to} {...linkProps} />;
        }
      ),
    [to]
  );
  return (
    <li>
      <ListItem button component={CustomLink}>
        <ListItemIcon>{icon}</ListItemIcon>
        <ListItemText primary={primary} />
      </ListItem>
    </li>
  );
}
```

## роутинг

Для react-router-dom и next-link есть готовые решения

# MUI - system

Основной метод кастомизации в MUI
Основывается на sx пропсе, позволяет избежать вложенности контекста и лишних компонентов
Минус заключается в том, что компоненты с sx самые затратные по производительности

SX пропс предоставляет все CSS свойства, а также кастомные. Поддерживает css – свойства, такие как hover, media-queries, вложенные селекторы
Box – основной элемент для кастомизации

Утилиты затрагивают такие аспекты как border, display, flex-box, grid, palette

Основные компоненты (у всех есть element, sx, children пропсы):

- Box (базовый div-компонент)
- Container (позволяет центровать горизонтально контент, fluid – проп позволяет растянуть на всю ширину, fixed – будет синхронизироваться с breakpoint), props { classes, disableGutters: boolean (уберет padding-и слева и справа), fixed, maxWidth: BreakPoints}
- Grid (основной элемент, который позволяет контролировать сетку), props {columns: number (количество колонок), columnSpacing, container, direction, disableEqualOverflow, lg, lgOffset … xs xsOffset (сколько колонок занимает / сколько колонок пропустить ), wrap, }
- Stack (вертикальная сетка) props {direction, divider: node, spacing, useFlexGap}

## breakpoints

вариант объектом

```js
sx={{
  width: {
    xs: 100, // theme.breakpoints.up('xs')
    sm: 200, // theme.breakpoints.up('sm')
    md: 300, // theme.breakpoints.up('md')
    lg: 400, // theme.breakpoints.up('lg')
    xl: 500, // theme.breakpoints.up('xl')
  },
}}

```

2. Вариант - массивом

```js
<Box sx={{ width: [100, 200, 300] }}>This box has a responsive width.</Box>
```

## sx-prop

border
Поддерживает только значения из темы
gap – по умолчанию поддерживается
palette – можно обращаться к palette
width – ½ === 50%, 20 === 20px

Поддерживает коллбек значения для учета темы

```tsx
<Box sx={{ height: (theme) => theme.spacing(10) }} />
```

Переопределение от внешних флагов, можно добавлять коллбеки

```tsx
<Box
  sx={[
    {
      "&:hover": {
        color: "red",
        backgroundColor: "white",
      },
    },
    foo && {
      // foo – некий флаг
      "&:hover": { backgroundColor: "grey" },
    },
    (theme) => ({
      "&:hover": {
        color: theme.palette.primary.main,
      },
    }),
  ]}
/>
```

## styled

styled(Component, [options])(styles) => Component

Component: The component that will be wrapped.

options:

- shouldForwardProp ((prop: string) => bool [optional])//должен пропс быть передан в компонент, таким образом можно отслеживать проброс дополнительных пропсов в элемент
- label (string [optional]): префикс для компоненты (root, label, value…) – поменяет name + slot
- name (string [optional]): определяет компонент в теме
- slot (string [optional]): префикс для компоненты (root, label, value…)
- overridesResolver ((props: object, styles: Record<string, styles>) => styles [optional])//функция для переопределения стилей. props – получает все props компонента (children и объект theme), styles – объект styleOverrides
- skipVariantsResolver (bool)// отключает функцию выше,
- skipSx (bool [optional])// отключает sx

SX
В styled нельзя использовать сокращенные названия как mx, mt и др.
Если указать в sx padding: 1//1px
Разница в использовании prop

```js
const MyStyledButton = styled("button")((props) => ({
  backgroundColor: props.myBackgroundColor,
}));
const MyStyledButton = (props) => (
  <Button sx={{ backgroundColor: props.myCustomColor }}>
    {props.children}
  </Button>
);
```

Можно использовать sx в styled через theme.unstable_sx

# BP

## imports

Импорты

```js
// 🐌 Named
import { Delete } from "@mui/icons-material";
// 🚀 Default
import Delete from "@mui/icons-material/Delete";
```

## props

sx проп менее производительный вариант, чем указание стилевых проп (системные UI-пропы)
Все значения темы доступны через styled, как параметр theme
