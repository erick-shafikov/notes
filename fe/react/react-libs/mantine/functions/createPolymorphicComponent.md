# createPolymorphicComponent

оборачивает компонент и добавляет ему поддержку component пропа с правильными типами

```ts
createPolymorphicComponent<DefaultElement, Props>(Component);
```

```tsx
import { forwardRef } from "react";
import {
  createPolymorphicComponent,
  Button,
  ButtonProps,
  Group,
} from "@mantine/core";

interface CustomButtonProps extends ButtonProps {
  label: string;
}

// Default root element is 'button', but it can be changed with 'component' prop
const CustomButton = createPolymorphicComponent<"button", CustomButtonProps>(
  forwardRef<HTMLButtonElement, CustomButtonProps>(
    ({ label, ...others }, ref) => (
      <Button {...others} ref={ref}>
        {label}
      </Button>
    )
  )
);

// Default root element is 'a', but it can be changed with 'component' prop
const CustomButtonAnchor = createPolymorphicComponent<"a", CustomButtonProps>(
  forwardRef<HTMLAnchorElement, CustomButtonProps>(
    ({ label, ...others }, ref) => (
      <Button component="a" {...others} ref={ref}>
        {label}
      </Button>
    )
  )
);

function Demo() {
  return (
    <Group>
      <CustomButton label="Button by default" color="cyan" />
      <CustomButtonAnchor
        label="Anchor by default"
        href="https://mantine.dev"
        target="_blank"
      />
    </Group>
  );
}
```

Кастомные компоненты

```tsx
import { forwardRef } from "react";
import {
  Box,
  BoxProps,
  createPolymorphicComponent,
  Group,
} from "@mantine/core";

interface MyButtonProps extends BoxProps {
  label: string;
}

const MyButton = createPolymorphicComponent<"button", MyButtonProps>(
  forwardRef<HTMLButtonElement, MyButtonProps>(({ label, ...others }, ref) => (
    <Box component="button" {...others} ref={ref}>
      {label}
    </Box>
  ))
);

function Demo() {
  return (
    <Group>
      <MyButton label="Button by default" />
      <MyButton
        label="MyButton as anchor"
        component="a"
        href="https://mantine.dev"
        target="_blank"
      />
    </Group>
  );
}
```
