# styled

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

```tsx
// пример на styled components
const StatWrapper = styled("div")(
  ({ theme }) => `
  background-color: ${theme.palette.background.paper};
  box-shadow: ${theme.shadows[1]};
  border-radius: ${theme.shape.borderRadius}px;
  padding: ${theme.spacing(2)};
  min-width: 300px;
`
);

const StatHeader = styled("div")(
  ({ theme }) => `
  color: ${theme.palette.text.secondary};
`
);

const StyledTrend = styled(TrendingUpIcon)(
  ({ theme }) => `
  color: ${theme.palette.success.dark};
  font-size: 16px;
  vertical-alignment: sub;
`
);

const StatValue = styled("div")(
  ({ theme }) => `
  color: ${theme.palette.text.primary};
  font-size: 34px;
  font-weight: ${theme.typography.fontWeightMedium};
`
);

const StatDiff = styled("div")(
  ({ theme }) => `
  color: ${theme.palette.success.dark};
  display: inline;
  font-weight: ${theme.typography.fontWeightMedium};
  margin-left: ${theme.spacing(0.5)};
  margin-right: ${theme.spacing(0.5)};
`
);

const StatPrevious = styled("div")(
  ({ theme }) => `
  color: ${theme.palette.text.secondary};
  display: inline;
  font-size: 12px;
`
);

export const StyledExample = () => {
  return (
    <StatWrapper>
      <StatHeader>Sessions</StatHeader>
      <StatValue>98.3 K</StatValue>
      <StyledTrend />
      <StatDiff>18.77%</StatDiff>
      <StatPrevious>vs last week</StatPrevious>
    </StatWrapper>
  );
};

//--------------------------------------------------------------------

export const MUIEx = () => {
  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        boxShadow: 1,
        borderRadius: 1,
        p: 2,
        minWidth: 300,
      }}
    >
      <Box sx={{ color: "text.secondary" }}>Sessions</Box>
      <Box sx={{ color: "text.primary", fontSize: 34, fontWeight: "medium" }}>
        98.3 K
      </Box>
      <Box
        component={TrendingUpIcon}
        sx={{ color: "success.dark", fontSize: 16, verticalAlign: "sub" }}
      />
      <Box
        sx={{
          color: "success.dark",
          display: "inline",
          fontWeight: "medium",
          mx: 0.5,
        }}
      >
        18.77%
      </Box>
      <Box sx={{ color: "text.secondary", display: "inline", fontSize: 12 }}>
        vs. last week
      </Box>
    </Box>
  );
};
```

Можно использовать sx в styled через theme.unstable_sx
