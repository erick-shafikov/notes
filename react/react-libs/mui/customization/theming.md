# theming

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

```tsx
import { createTheme, ThemeProvider } from "@mui/material/styles";
import Button from "@mui/material/Button";
import {
  Box,
  Chip,
  // ComponentsOverrides,
  // ComponentsProps,
  // ComponentsPropsList,
  // Interpolation,
  Slider,
  sliderClasses,
  // Theme,
} from "@mui/material";
import { Check } from "@mui/icons-material";
// import { blue } from "@mui/material/colors";

// ----------------------------------------------------------------------
// Изменение значений по умолчанию компонентов
const compTheme = createTheme({
  components: {
    // Name of the component ⚛️
    MuiButtonBase: {
      defaultProps: {
        // The default props to change
        disableRipple: true, // No more ripple, on the whole application 💣!
      },
    },
  },
});

function DefaultProps() {
  return (
    <ThemeProvider theme={compTheme}>
      <Button>This button has disabled ripples.</Button>
    </ThemeProvider>
  );
}

// ----------------------------------------------------------------------
// Изменение значений компонентов
const globalTheme = createTheme({
  components: {
    // имя компонента
    MuiButton: {
      styleOverrides: {
        // имя слота
        root: {
          //переопределенное значение
          fontSize: "1rem",
        },
      },
    },
  },
});

function GlobalThemeOverride() {
  return (
    <ThemeProvider theme={globalTheme}>
      <Button>font-size: 1rem</Button>
    </ThemeProvider>
  );
}

// ----------------------------------------------------------------------
// Значения базированные на стандартных атрибутах (пропах) с помощью styleOverrides

const styleOverridesTheme = createTheme({
  components: {
    MuiSlider: {
      //объект для переопределения, в данном случает динамически переопределяем valueLabel
      styleOverrides: {
        valueLabel: ({ ownerState, theme }) => ({
          ...(ownerState.orientation === "vertical" && {
            backgroundColor: "transparent",
            color: theme.palette.grey[500],
            fontWeight: 700,
            padding: 0,
            left: "3rem",
          }),
          [`&.${sliderClasses.valueLabelOpen}`]: {
            transform: "none",
            top: "initial",
          },
        }),
      },
    },
  },
});

function valuetext(value: number) {
  return `${value}°C`;
}

function GlobalThemeOverrideCallback() {
  return (
    <ThemeProvider theme={styleOverridesTheme}>
      <Box sx={{ height: 180, display: "inline-block" }}>
        <Slider
          getAriaLabel={() => "Temperature"}
          orientation="vertical"
          getAriaValueText={valuetext}
          defaultValue={[25, 50]}
          marks={[
            { value: 0 },
            { value: 25 },
            { value: 50 },
            { value: 75 },
            { value: 100 },
          ]}
          valueLabelFormat={valuetext}
          valueLabelDisplay="on"
        />
      </Box>
    </ThemeProvider>
  );
}
// ----------------------------------------------------------------------
// Переопределение через sx - prop

const sxTheme = createTheme({
  components: {
    MuiChip: {
      styleOverrides: {
        root: ({ theme }) =>
          theme.unstable_sx({
            px: 1,
            py: 0.25,
            borderRadius: 1, // 4px as default.
          }),
        label: {
          padding: "initial",
        },
        icon: ({ theme }) =>
          theme.unstable_sx({
            mr: 0.5,
            ml: "-2px",
          }),
      },
    },
  },
});

function GlobalThemeOverrideSx() {
  return (
    <ThemeProvider theme={sxTheme}>
      <Chip
        color="success"
        label={
          <span>
            <b>Status:</b> Completed
          </span>
        }
        icon={<Check fontSize="small" />}
      />
    </ThemeProvider>
  );
}
// ----------------------------------------------------------------------
//Новые варианты компонентов

declare module "@mui/material/Button" {
  interface ButtonPropsVariantOverrides {
    dashed: true; //типизируем новый вариант
  }
}

// работает только если в @mui/material/styles/variants удалить или изменить называние типа
// declare module "@mui/material/styles/variants" {
//   type ComponentsVariants = {
//     [Name in keyof ComponentsPropsList]?: Array<{
//       props:
//         | Partial<ComponentsPropsList[Name]>
//         | ((props: Partial<ComponentsPropsList[Name]>) => boolean);
//       style: Interpolation<{ theme: Theme }>;
//     }>;
//   };
// }

const defaultTheme = createTheme();

const newVariantComponent = createTheme({
  components: {
    MuiButton: {
      // варианты в зависимости от Props, добавляем dashed компонент
      variants: [
        {
          props: { variant: "dashed" },
          style: {
            textTransform: "none",
            border: `2px dashed ${defaultTheme.palette.primary.main}`,
            color: defaultTheme.palette.primary.main,
          },
        },
        {
          props: { variant: "dashed", color: "secondary" },
          style: {
            border: `2px dashed ${defaultTheme.palette.secondary.main}`,
            color: defaultTheme.palette.secondary.main,
          },
        },
        {
          props: { variant: "dashed", size: "large" },
          style: {
            borderWidth: 4,
          },
        },
        {
          props: { variant: "dashed", color: "secondary", size: "large" },
          style: {
            fontSize: 18,
          },
        },
        // Вариант с cb проблемы с типизацией
        // {
        //   props: (props) =>
        //     props.variant === "dashed" && props.color !== "secondary",
        //   style: {
        //     textTransform: "none",
        //     border: `2px dashed ${blue[500]}`,
        //   },
        // },
      ],
    },
  },
});

// ----------------------------------------------------------------------
//Переписать с помощью переопределения токенов темы
/* const themeConfVariables = createTheme({
  typography: {
    button: {
      fontSize: "1rem",
    },
  },
}); */

function GlobalThemeVariants() {
  return (
    <ThemeProvider theme={newVariantComponent}>
      <Button variant="dashed" sx={{ m: 1 }}>
        Dashed
      </Button>
      <Button variant="dashed" color="secondary" sx={{ m: 1 }}>
        Secondary
      </Button>
      <Button variant="dashed" size="large" sx={{ m: 1 }}>
        Large
      </Button>
      <Button variant="dashed" color="secondary" size="large" sx={{ m: 1 }}>
        Secondary large
      </Button>
    </ThemeProvider>
  );
}

const Components = () => {
  return (
    <>
      <DefaultProps />
      <GlobalThemeOverride />
      <GlobalThemeOverrideCallback />
      <GlobalThemeOverrideSx />
      <GlobalThemeVariants />
    </>
  );
};

export default Components;
```

# dark mode

Запустить проект по умолчанию в режиме темной темы

```js
const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});
```

Позволяет узнать тему системы, позволяет узнать не только значения palette, но и другие

```tsx
import CssBaseline from "@mui/material/CssBaseline";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <main>This app is using the dark mode</main>
    </ThemeProvider>
  );
}
// ----------------------------------------------------------------------
// Пример компонента переключателя темы
import * as React from "react";
import IconButton from "@mui/material/IconButton";
import Box from "@mui/material/Box";
import { useTheme, ThemeProvider, createTheme } from "@mui/material/styles";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import Brightness7Icon from "@mui/icons-material/Brightness7";

//Контекст, переключатель темы
const ColorModeContext = React.createContext({ toggleColorMode: () => {} });

// Компонент
function MyApp() {
  const theme = useTheme();
  //Получаем тему, в которой лежит переключатель темы
  const colorMode = React.useContext(ColorModeContext);
  return (
    <Box
      sx={{
        display: "flex",
        width: "100%",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "background.default",
        color: "text.primary",
        borderRadius: 1,
        p: 3,
      }}
    >
      {theme.palette.mode} mode
      <IconButton
        sx={{ ml: 1 }}
        onClick={colorMode.toggleColorMode}
        color="inherit"
      >
        {theme.palette.mode === "dark" ? (
          <Brightness7Icon />
        ) : (
          <Brightness4Icon />
        )}
      </IconButton>
    </Box>
  );
}

//Оболочка
function ToggleColorMode() {
  // тема определяется состоянием
  const [mode, setMode] = React.useState<"light" | "dark">("light");

  // мемоизируем объект контекста переключателя
  const colorMode = React.useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === "light" ? "dark" : "light"));
      },
    }),
    []
  );

  // мемоизируем тему
  const theme = React.useMemo(
    () =>
      createTheme({
        palette: {
          mode,
        },
      }),
    [mode]
  );

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <MyApp />
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

// ----------------------------------------------------------------------
// Экспорт
const DarkMode = () => {
  return (
    <>
      <App />
      <ToggleColorMode />
    </>
  );
};

export default DarkMode;
```

# вложенные темы

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

# компоненты с учетом темой

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

```tsx
import Checkbox from "@mui/material/Checkbox";
import { createTheme, ThemeProvider, styled } from "@mui/material/styles";
import { green } from "@mui/material/colors";

declare module "@mui/material/styles" {
  interface Theme {
    status: {
      danger: string;
    };
  }
  // allow configuration using `createTheme`
  interface ThemeOptions {
    status?: {
      danger?: string;
    };
  }
}

const CustomCheckbox = styled(Checkbox)(({ theme }) => ({
  color: theme.status.danger,
  "&.Mui-checked": {
    color: theme.status.danger,
  },
}));

const theme = createTheme({
  status: {
    danger: green[500],
  },
});

function CustomStyles() {
  return (
    <ThemeProvider theme={theme}>
      <CustomCheckbox defaultChecked />
    </ThemeProvider>
  );
}

const Theming = () => {
  return (
    <div>
      <CustomStyles />
    </div>
  );
};

export default Theming;
```

# palette

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

```tsx
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { blue } from "@mui/material/colors";
import { Box, Stack } from "@mui/system";
import { Typography } from "@mui/material";
// ----------------------------------------------------------------------
// Пример добавления darker-токена в объект темы
declare module "@mui/material/styles" {
  interface PaletteColor {
    darker?: string;
  }

  interface SimplePaletteColorOptions {
    darker?: string;
  }
}

const theme = createTheme({
  palette: {
    primary: {
      light: blue[300],
      main: blue[500],
      dark: blue[700],
      darker: blue[900],
    },
  },
});

function AddingColorTokens() {
  return (
    <ThemeProvider theme={theme}>
      <Stack direction="row" gap={1}>
        <Stack alignItems="center">
          <Typography variant="body2">light</Typography>
          <Box sx={{ bgcolor: `primary.light`, width: 40, height: 20 }} />
        </Stack>
        <Stack alignItems="center">
          <Typography variant="body2">main</Typography>
          <Box sx={{ bgcolor: `primary.main`, width: 40, height: 20 }} />
        </Stack>
        <Stack alignItems="center">
          <Typography variant="body2">dark</Typography>
          <Box sx={{ bgcolor: `primary.dark`, width: 40, height: 20 }} />
        </Stack>
        <Stack alignItems="center">
          <Typography variant="body2">darker</Typography>
          <Box sx={{ bgcolor: `primary.darker`, width: 40, height: 20 }} />
        </Stack>
      </Stack>
    </ThemeProvider>
  );
}

// ----------------------------------------------------------------------

const Palette = () => {
  return <AddingColorTokens />;
};

export default Palette;
```

# typography

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

```tsx
import { Box, CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import RalewayWoff2 from "./fonts/Raleway/static/Raleway-Regular.ttf";
// ----------------------------------------------------------------------
const LocalFonts = () => {
  const theme = createTheme({
    typography: {
      fontFamily: "Raleway, Arial",
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: `
          @font-face {
            font-family: 'Raleway';
            font-style: normal;
            font-display: swap;
            font-weight: 400;
            src: local('Raleway'), local('Raleway-Regular'), url(${RalewayWoff2}) format('woff2');
            unicodeRange: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF;
          }
        `,
      },
    },
  });
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          fontFamily: "Raleway",
        }}
      >
        Raleway
      </Box>
    </ThemeProvider>
  );
};
// ----------------------------------------------------------------------

const RemoteFonts = () => {
  //в index.html добавлена ссылка
  /* 
  <link
      href="https://fonts.googleapis.com/css2?family=Yellowtail&display=swap"
      rel="stylesheet"
    />
*/
  const theme = createTheme({
    typography: { fontFamily: ["Yellowtail", "cursive"].join(",") },
  });
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          fontFamily: "Yellowtail",
        }}
      >
        Raleway
      </Box>
    </ThemeProvider>
  );
};

const Typography = () => {
  return (
    <>
      <RemoteFonts />
      <LocalFonts />
    </>
  );
};

export default Typography;
```

# spacing

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

# breakpoints

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

```tsx
import { styled } from "@mui/material/styles";
import Typography from "@mui/material/Typography";
import { red, green, blue } from "@mui/material/colors";

const Root = styled("div")(({ theme }) => ({
  padding: theme.spacing(1),
  [theme.breakpoints.down("md")]: {
    backgroundColor: red[500],
  },
  [theme.breakpoints.up("md")]: {
    backgroundColor: blue[500],
  },
  [theme.breakpoints.not("md")]: {
    backgroundColor: "red",
  },
  [theme.breakpoints.up("lg")]: {
    backgroundColor: green[500],
  },
  [theme.breakpoints.between("sm", "md")]: {
    backgroundColor: "red",
  },
}));

function MediaQuery() {
  return (
    <Root>
      <Typography>down(md): red</Typography>
      <Typography>up(md): blue</Typography>
      <Typography>up(lg): green</Typography>
    </Root>
  );
}

const Breakpoints = () => {
  return (
    <>
      <MediaQuery />
    </>
  );
};

export default Breakpoints;
```

# density

Компоненты в который плотность идет пропсом

Button, Fab, FilledInput, FormControl, FormHelperText, IconButton, InputBase, InputLabel, ListItem, OutlinedInput, Table, TextField, Toolbar

# transition

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

```tsx
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import { deepPurple } from "@mui/material/colors";
import Avatar from "@mui/material/Avatar";

const customTheme = createTheme({
  palette: {
    primary: {
      main: deepPurple[500],
    },
  },
});

const StyledAvatar = styled(Avatar)`
  ${({ theme }) => `
  cursor: pointer;
  background-color: ${theme.palette.primary.main};
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

function TransitionHover() {
  return (
    <ThemeProvider theme={customTheme}>
      <StyledAvatar>OP</StyledAvatar>
    </ThemeProvider>
  );
}

const Transition = () => {
  return <TransitionHover />;
};

export default Transition;
```
