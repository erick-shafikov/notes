## кастомизация

Кастомизация компонентов:

- Одноразовая кастомизация, точечная – можно производить с помощью sx пропса, в нем можно указывать измененные стили для вложенных компонентов
- Переисользуемые компоненты – с помощью styled можно изменять компоненты
- Глобальные темы – специальный компонент

```tsx
import Slider, { SliderProps } from "@mui/material/Slider";
import { alpha, styled } from "@mui/material/styles";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";

// ----------------------------------------------------------------------
//изменение с помощью sx пропса

function OneOffCustomization() {
  return (
    <Slider
      defaultValue={30}
      sx={{
        width: 300,
        color: "success.main",
        "& .MuiSlider-thumb": {
          //переопределение вложенных элементов
          borderRadius: "1px",
        },
      }}
    />
  );
}
// ----------------------------------------------------------------------
//изменение с помощью classNames

import "./style.css";
import { MenuItem } from "@mui/material";
import { useState } from "react";

const ClassNamesCustomization = () => (
  <MenuItem selected className="MenuItem">
    Menu item
  </MenuItem>
);
// ----------------------------------------------------------------------
//переисользуемые компоненты

const SuccessSlider = styled(Slider)<SliderProps>(({ theme }) => ({
  width: 300,
  color: theme.palette.success.main,
  "& .MuiSlider-thumb": {
    "&:hover, &.Mui-focusVisible": {
      boxShadow: `0px 0px 0px 8px ${alpha(theme.palette.success.main, 0.16)}`,
    },
    "&.Mui-active": {
      boxShadow: `0px 0px 0px 14px ${alpha(theme.palette.success.main, 0.16)}`,
    },
  },
}));

function ReusableComponent() {
  return <SuccessSlider defaultValue={30} />;
}

//переисользуемые компоненты c пропсами

interface StyledSliderProps extends SliderProps {
  success?: boolean;
}

const StyledSlider = styled(Slider, {
  shouldForwardProp: (prop) => prop !== "success",
})<StyledSliderProps>(({ success, theme }) => ({
  width: 300,
  ...(success && {
    color: theme.palette.success.main,
    "& .MuiSlider-thumb": {
      [`&:hover, &.Mui-focusVisible`]: {
        boxShadow: `0px 0px 0px 8px ${alpha(theme.palette.success.main, 0.16)}`,
      },
      [`&.Mui-active`]: {
        boxShadow: `0px 0px 0px 14px ${alpha(
          theme.palette.success.main,
          0.16
        )}`,
      },
    },
  }),
}));

function ReusableComponentDynamic() {
  const [success, setSuccess] = useState(false);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSuccess(event.target.checked);
  };

  return (
    <>
      <FormControlLabel
        control={
          <Switch
            checked={success}
            onChange={handleChange}
            color="primary"
            value="dynamic-class-name"
          />
        }
        label="Success"
      />
      <StyledSlider success={success} defaultValue={30} sx={{ mt: 1 }} />
    </>
  );
}

const CustomSlider = styled(Slider)({
  width: 300,
  color: "var(--color)",
  "& .MuiSlider-thumb": {
    [`&:hover, &.Mui-focusVisible`]: {
      boxShadow: "0px 0px 0px 8px var(--box-shadow)",
    },
    [`&.Mui-active`]: {
      boxShadow: "0px 0px 0px 14px var(--box-shadow)",
    },
  },
});

const successVars = {
  "--color": "#4caf50",
  "--box-shadow": "rgb(76, 175, 80, .16)",
} as React.CSSProperties;

const defaultVars = {
  "--color": "#1976d2",
  "--box-shadow": "rgb(25, 118, 210, .16)",
} as React.CSSProperties;

function ReusableComponentStyleObj() {
  const [vars, setVars] = useState<React.CSSProperties>(defaultVars);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setVars(event.target.checked ? successVars : defaultVars);
  };

  return (
    <>
      <FormControlLabel
        control={
          <Switch
            checked={vars === successVars}
            onChange={handleChange}
            color="primary"
            value="dynamic-class-name"
          />
        }
        label="Success"
      />
      <CustomSlider _style={vars} defaultValue={30} sx={{ mt: 1 }} />
    </>
  );
}
```
