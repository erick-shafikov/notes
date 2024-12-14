```tsx
import { styled, createTheme, ThemeProvider } from "@mui/system";

// ----------------------------------------------------------------------
// Basic usage
const MyComponent = styled("div")({
  color: "darkslategray",
  backgroundColor: "aliceblue",
  padding: 8,
  borderRadius: 4,
});

function BasicUsage() {
  return <MyComponent>Styled div</MyComponent>;
}

// ----------------------------------------------------------------------
// Basic usage with theme

const customBasicTheme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
      contrastText: "white",
    },
  },
});

const MyThemeComponent = styled("div")(({ theme }) => ({
  color: theme.palette.primary.contrastText,
  backgroundColor: theme.palette.primary.main,
  padding: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
}));

function BasicThemeUsage() {
  return (
    <ThemeProvider theme={customBasicTheme}>
      <MyThemeComponent>Styled div with theme</MyThemeComponent>
    </ThemeProvider>
  );
}

// ----------------------------------------------------------------------
// Custom components
interface MyCustomThemeComponentProps {
  color?: "primary" | "secondary";
  variant?: "normal" | "dashed";
}

const customTheme = createTheme({
  components: {
    MyThemeComponent: {
      styleOverrides: {
        root: {
          color: "darkslategray",
        },
        primary: {
          color: "darkblue",
        },
        secondary: {
          color: "darkred",
          backgroundColor: "pink",
        },
      },
      variants: [
        {
          props: { variant: "dashed", color: "primary" },
          style: {
            border: "1px dashed darkblue",
          },
        },
        {
          props: { variant: "dashed", color: "secondary" },
          style: {
            border: "1px dashed darkred",
          },
        },
      ],
    },
  },
});

const MyCustomThemeComponent = styled("div", {
  // Configure which props should be forwarded on DOM
  shouldForwardProp: (prop) =>
    prop !== "color" && prop !== "variant" && prop !== "sx",
  name: "MyThemeComponent",
  slot: "Root",
  // We are specifying here how the styleOverrides are being applied based on props
  overridesResolver: (props, styles) => {
    // console.log("props:", props, "styles:", styles);
    // console.log("props:", props);
    console.log("styles:", styles);

    return [
      styles.root,
      props.color === "primary" && styles.primary,
      props.color === "secondary" && styles.secondary,
    ];
  },
  //
  // skipVariantsResolver: true,
  // skipSx: true,
})<MyCustomThemeComponentProps>(({ theme }) => ({
  backgroundColor: "aliceblue",
  padding: theme.spacing(1),
}));

function UsingOptions() {
  return (
    <ThemeProvider theme={customTheme}>
      {/* <MyCustomThemeComponent sx={{ m: 1 }} color="primary" variant="dashed">
        Primary
      </MyCustomThemeComponent> */}
      <MyCustomThemeComponent sx={{ m: 1 }} color="secondary">
        Secondary
      </MyCustomThemeComponent>
    </ThemeProvider>
  );
}

const Styled = () => {
  return (
    <>
      <BasicUsage />
      <BasicThemeUsage />
      <UsingOptions />
    </>
  );
};

export default Styled;
```

how to guides

```tsx
import * as React from "react";
import Stack from "@mui/material/Stack";
import { styled, useThemeProps } from "@mui/material/styles";

export interface StatProps {
  //типы для расширения StatOwnerState
  value: number | string;
  unit: string;
  variant?: "outlined";
}

interface StatOwnerState extends StatProps {
  //пары ключ-значение, для внутреннего состояния, которыми будет стилизован slot, но невиден пользователю
}

// стилизация root - компонента
const StatRoot = styled("div", {
  name: "MuiStat", //название компонента
  slot: "root", //наименование слота
  // shouldForwardProp: (prop) => {
  //   console.log(prop);
  //   return true;
  // },
  // label: "stat-root",
  // overridesResolver: (...args) => {
  //   console.log(args);

  //   return {};
  // },

  // skipVariantsResolver: true,
})<{ ownerState: StatOwnerState }>(({ theme, ownerState }) => ({
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(0.5),
  padding: theme.spacing(3, 4),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[2],
  letterSpacing: "-0.025em",
  fontWeight: 600,
  ...(ownerState.variant === "outlined" && {
    // обработка outlined - варианта
    border: `2px solid ${theme.palette.divider}`,
    boxShadow: "none",
  }),
}));

// стилизация value - компонента
const StatValue = styled("div", {
  name: "MuiStat",
  slot: "value",
})<{ ownerState: StatOwnerState }>(({ theme }) => ({
  ...theme.typography.h3,
}));

//стилизация unit - компонента
const StatUnit = styled("div", {
  name: "MuiStat",
  slot: "unit",
})<{ ownerState: StatOwnerState }>(({ theme }) => ({
  ...theme.typography.body2,
  color: theme.palette.text.secondary,
}));

//формирование компонента
const Stat = React.forwardRef<HTMLDivElement, StatProps>(function Stat(
  inProps,
  ref
) {
  const props = useThemeProps({ props: inProps, name: "MuiStat" }); //получаем значения по умолчанию, передаем props и имя
  const { value, unit, variant, ...other } = props;

  const ownerState = { ...props, variant };

  return (
    <StatRoot ref={ref} ownerState={ownerState} {...other}>
      <StatValue ownerState={ownerState}>{value}</StatValue>
      <StatUnit ownerState={ownerState}>{unit}</StatUnit>
    </StatRoot>
  );
});

export default function StatFullTemplate() {
  return (
    <Stack direction="row" spacing={2}>
      <Stat value="1.9M" unit="Favorites" />
      <Stat value="5.1M" unit="Views" variant="outlined" />
    </Stack>
  );
}
```
