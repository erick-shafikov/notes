# sx-prop

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

Смысл в том, что можно сделать все компоненты на Box - элементе

```tsx
// пример
sx={{
    boxShadow: 1, // theme.shadows[1]
    color: 'primary.main', // theme.palette.primary.main
    m: 1, // margin: theme.spacing(1)
    p: {
      xs: 1, // [theme.breakpoints.up('xs')]: { padding: theme.spacing(1) }
    },
    zIndex: 'tooltip', // theme.zIndex.tooltip
    ":hover": {
      boxShadow: 6,
    },
    // медиазапросы
    '@media print': {
      width: 300,
    },
    // вложенность
    '& .ChildSelector': {
      bgcolor: 'primary.main',
    },
  }}

  sx={{ color: "text.secondary" }} - обращение к палете

// spacing
  const theme = {
  spacing: 8,
}
// 1 === 8px


<Box sx={{ m: -2 }} /> // margin: -16px;
<Box sx={{ m: 0 }} /> // margin: 0px;
<Box sx={{ m: 0.5 }} /> // margin: 4px;
<Box sx={{ m: 2 }} /> // margin: 16px;
```

```tsx
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import { styled } from "@mui/material";
import Box from "@mui/material/Box";

//--------------------------------------------------------------------
// пример на sx-пропсе
export function Why() {
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
}
```

```tsx
//--------------------------------------------------------------------
// пример компонента
import { alpha } from "@mui/material/styles";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";

export function Demo() {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: { xs: "column", md: "row" },
        alignItems: "center",
        bgcolor: "background.paper",
        overflow: "hidden",
        borderRadius: "12px",
        boxShadow: 1,
        fontWeight: "bold",
      }}
    >
      <Box
        component="img"
        sx={{
          height: 233,
          width: 350,
          maxHeight: { xs: 233, md: 167 },
          maxWidth: { xs: 350, md: 250 },
        }}
        alt="The house from the offer."
        src="https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&w=350&dpr=2"
      />
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: { xs: "center", md: "flex-start" },
          m: 3,
          minWidth: { md: 350 },
        }}
      >
        <Box component="span" sx={{ fontSize: 16, mt: 1 }}>
          123 Main St, Phoenix AZ
        </Box>
        <Box component="span" sx={{ color: "primary.main", fontSize: 22 }}>
          $280,000 — $310,000
        </Box>
        <Box
          sx={{
            mt: 1.5,
            p: 0.5,
            backgroundColor: (theme) => alpha(theme.palette.primary.main, 0.1),
            borderRadius: "5px",
            color: "primary.main",
            fontWeight: "medium",
            display: "flex",
            fontSize: 12,
            alignItems: "center",
            "& svg": {
              fontSize: 21,
              mr: 0.5,
            },
          }}
        >
          <ErrorOutlineIcon />
          CONFIDENCE SCORE 85%
        </Box>
      </Box>
    </Box>
  );
}
```
