# HOC

```tsx
import { MouseEventHandler, useState } from "react";
const App = ({
  darkTheme,
  toggleTheme,
}: {
  darkTheme: boolean;
  toggleTheme: MouseEventHandler<HTMLDivElement>;
}) => {
  return (
    <div data-theme={darkTheme ? "dark" : "light"} onClick={toggleTheme} />
  );
};
const withTheme = (Component: any) => {
  function Func(props: any) {
    const [darkTheme, setDarkTheme] = useState(true);
    return (
      <Component
        {...props}
        darkTheme={darkTheme}
        toggleTheme={() => setDarkTheme(!darkTheme)}
      />
    );
  }
  return Func;
};
export default withTheme(App);
```
