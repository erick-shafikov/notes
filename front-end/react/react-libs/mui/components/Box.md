/* 
  Box - универсальный компонент для группировки других элементов 
  props: sx

  createBox() - для box, который вне текущей темы
  */

import { Box, ThemeProvider, createBox, createTheme } from "@mui/system";

// доступ ко всем свойствам темы
export function BoxSx() {
  return (
    <ThemeProvider
      theme={{
        palette: {
          primary: {
            main: "#007FFF",
            dark: "#0066CC",
          },
        },
      }}
    >
      <Box
        sx={{
          width: 100,
          height: 100,
          borderRadius: 1,
          bgcolor: "primary.main",
          "&:hover": {
            bgcolor: "primary.dark",
          },
        }}
      />
    </ThemeProvider>
  );
}

const defaultTheme = createTheme({
  // your custom theme values
});

const MyBox = createBox({ defaultTheme });

export default MyBox;
