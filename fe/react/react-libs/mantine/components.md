# withProps

каждый компонент имеет withProps метод

```tsx
import { IMaskInput } from "react-imask";
import { Button, InputBase } from "@mantine/core";

const LinkButton = Button.withProps({
  component: "a",
  target: "_blank",
  rel: "noreferrer",
  variant: "subtle",
});

const PhoneInput = InputBase.withProps({
  mask: "+7 (000) 000-0000",
  component: IMaskInput,
  label: "Your phone number",
  placeholder: "Your phone number",
});

const Demo = () => (
  <>
    <LinkButton href="">Mantine website</LinkButton>
    <PhoneInput placeholder="Personal phone" />
  </>
);
```

# полиморфные компоненты

- проп component - принимает имя тега
- renderRoot prop или generic компоненты
- [функция createPolymorphicComponent позволяет создавать фабрику полиморфных компонентов](./functions/createPolymorphicComponent.md)

```tsx
//проп component
// модно пробрасывать и Link из библиотек
const X = () => <Button component="a">Mantine website</Button>;
```

```tsx
// renderRoot prop
const X = () => (
  <>
    <Button
      renderRoot={(props) => (
        <a href="https://mantine.dev/" target="_blank" {...props} />
      )}
    >
      Mantine website
    </Button>
  </>
);
```

Пример компонента оборачиваемого с ref, так как ref должен быть определенного HTMLElement - типа

```tsx
import { forwardRef } from "react";
import { Button, ButtonProps } from "@mantine/core";

interface LinkButtonProps
  extends ButtonProps,
    // из a убрать button-пропсы, что бы была совместимость с button
    Omit<React.ComponentPropsWithoutRef<"a">, keyof ButtonProps> {}

const LinkButton = forwardRef<HTMLAnchorElement, LinkButtonProps>(
  (props, ref) => <Button {...props} ref={ref} component="a" />
);

const X = () => (
  <LinkButton href="https://mantine.dev" target="_blank">
    Mantine website
  </LinkButton>
);
```
