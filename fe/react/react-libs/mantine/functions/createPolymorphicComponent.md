# createPolymorphicComponent

оборачивает компонент и добавляет ему поддержку component пропа с правильными типами, основная задача - фабрика полиморфных компонентов с поддержкой ts

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

// объект по умолчанию - button, но можно исправить с помощью пропса component
const CustomButton = createPolymorphicComponent<"button", CustomButtonProps>(
  forwardRef<HTMLButtonElement, CustomButtonProps>(
    ({ label, ...others }, ref) => (
      <Button {...others} ref={ref}>
        {label}
      </Button>
    )
  )
);

// здесь компонент по умолчанию - a
const CustomButtonAnchor = createPolymorphicComponent<"a", CustomButtonProps>(
  forwardRef<HTMLAnchorElement, CustomButtonProps>(
    ({ label, ...others }, ref) => (
      <Button component="a" {...others} ref={ref}>
        {label}
      </Button>
    )
  )
);

const X = () => (
  <>
    {/* кнопка */}
    <CustomButton label="Button by default" color="cyan" />
    {/* ссылка */}
    <CustomButtonAnchor
      label="Anchor by default"
      href="https://mantine.dev"
      target="_blank"
    />
  </>
);
```

# динамически определяемые компоненты

```tsx
import { Box } from "@mantine/core";

function KeepTypes() {
  return (
    <Box<"input"> component={(Math.random() > 0.5 ? "input" : "div") as any} />
  );
}

const x = () => <Box<any> component={Math.random() > 0.5 ? "input" : "div"} />;
```

# Преобразование компонентов mantine в полиморфные

```tsx
import { createPolymorphicComponent, Group, GroupProps } from "@mantine/core";

const PolymorphicGroup = createPolymorphicComponent<"button", GroupProps>(
  // не полиморфный компонент
  Group
);

const X = () => <PolymorphicGroup component="a" href="https://mantine.dev" />;
```
