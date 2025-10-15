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
