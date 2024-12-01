# Styled components

- !!! Создавать компоненты нужно только вне компонентов

# пропсы

```tsx
const Button = styled.button<{ $primary?: boolean }>`
  background: ${(props) => (props.$primary ? "#BF4F74" : "white")};`;

const Thing = styled.div.attrs((/* props */) => ({ tabIndex: 0 }))``; //задаем дефолтные атрибуты без пропсов

const Input = styled.input.attrs<{ $size?: string }>((props) => ({
  type: "text", //атрибуты стандартного Input
  $size: props.$size || "1em",
}))`  
margin: ${(props) => props.$size};
  `;

//пример работы с объектом
const PropsBox = styled.div<{ $background: string }>((props) => ({
  background: props.$background,
  height: "50px",
  width: "50px",
}));
```

# темы

```tsx
import { ThemeProvider } from "styled-components";
import { StyledComponentsPractice } from "./StyledComponentsPractice";
//создаем объект темы
const theme = {
  main: "mediumseagreen",
};
//прокидываем
export const StyledComponentsProvider = () => {
  return (
    <ThemeProvider theme={theme}>
      <StyledComponentsPractice />
    </ThemeProvider>
  );
};
//используем
const TestComponent = styled.input.attrs<{ $size?: string }>((props) => ({}))`
  background-color: ${(props) => props.theme.main}; //используем
`;
export const StyledComponentsPractice = () => {
  const theme = useTheme(); //можно обратится к теме с помощью специального хука
  console.log(theme);
  return <TestComponent />;
};
```

```ts
//_d.ts
//определяем типы для TS
import "styled-components"; // and extend them!
import theme from "@/styles/theme"; // где лежит объект темы

type Theme = typeof theme;

declare module "styled-components" {
  export interface DefaultTheme extends Theme {}
}
```

# Глобальные стили

createGlobalStyle - позволяет создать обертку для глобальных стилей

```ts
import { createGlobalStyle } from "styled-components";

export const GlobalStyles = createGlobalStyle`
  *,
  *::after,
  *::before {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
`;
// применение

export const App = () => {
  return (
    <>
      <GlobalStyles />
    </>
  );
};
```

# наследование стилей

```tsx
const Button = styled.button``;

const TomatoButton = styled(Button)``;

const Link = ({ className, children }) => (
  <a className={className}>{children}</a>
);

const StyledLink = styled(Link)`
  color: #bf4f74;
  font-weight: bold;
`;
```
