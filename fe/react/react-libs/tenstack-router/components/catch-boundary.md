# CatchBoundary

ловит ошибки от детей

```tsx
import { CatchBoundary } from "@tanstack/react-router";

function Component() {
  return (
    <CatchBoundary
      getResetKey={() => {
        "reset";
        //функция возвращает строку для сброса состояния компонента
      }}
      onCatch={(error) => {
        console.error(error);
        //функция вызываемая с компонентом ошибки
      }}
      errorComponent={<>компонент ошибки</>}
    >
      <div>My Component</div>
    </CatchBoundary>
  );
}
```
