# useMantineThemeHook

```jsx
import { useMantineTheme } from "@mantine/core";

function Demo() {
  const theme = useMantineTheme();

  return (
    <div
      style={{
        backgroundColor: theme.colors.blue[1],
        color: theme.colors.blue[9],
      }}
    >
      This is a blue theme
    </div>
  );
}
```
