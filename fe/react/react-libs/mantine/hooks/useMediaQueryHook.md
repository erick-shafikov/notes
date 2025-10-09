# useMediaQueryHook

```tsx
import { Tooltip, Button, em } from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";

function Demo() {
  const isMobile = useMediaQuery(`(max-width: ${em(750)})`);

  return (
    <Tooltip label={isMobile ? "Mobile" : "Desktop"}>
      <Button>Hover me</Button>
    </Tooltip>
  );
}
```
