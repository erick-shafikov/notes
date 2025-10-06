# useProps

Позволяет добавить кастомный компонент

```tsx
import { useProps, MantineThemeProvider, createTheme } from "@mantine/core";

interface CustomComponentProps {
  color?: string;
  children?: React.ReactNode;
}

const defaultProps = {
  color: "red",
} satisfies Partial<CustomComponentProps>;

function CustomComponent(props: CustomComponentProps) {
  const { color, children } = useProps("CustomComponent", defaultProps, props);
  return <div _style={{ color }}>{children}</div>;
}

const theme = createTheme({
  components: {
    CustomComponent: {
      defaultProps: {
        color: "green",
      },
    },
  },
});

function Demo() {
  return (
    <div>
      <CustomComponent>Default color</CustomComponent>

      <MantineThemeProvider theme={theme}>
        <CustomComponent>Provider color</CustomComponent>
        <CustomComponent color="blue">Prop color</CustomComponent>
      </MantineThemeProvider>
    </div>
  );
}
```
