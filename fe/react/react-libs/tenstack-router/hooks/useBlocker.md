Позволяет осуществить запрет на переход

```tsx
import { useBlocker } from "@tanstack/react-router";

function MyComponent() {
  const [formIsDirty, setFormIsDirty] = useState(false);

  //коллбеки на выполнение перехода
  const { proceed, reset, status } = useBlocker({
    // shouldBlockFn должна возвращать true - разрешить переход или false - не разрешать
    shouldBlockFn: ({ current, next }) => {
      //пример реализации с помощью промиса
      const shouldBlock = new Promise<boolean>((resolve) => {
        modals.open({
          title: "Are you sure you want to leave?",
          children: (
            <SaveBlocker
              confirm={() => {
                modals.closeAll();
                // разрешение промиса
                resolve(false);
              }}
              reject={() => {
                modals.closeAll();
                // разрешение промиса
                resolve(true);
              }}
            />
          ),
          onClose: () => resolve(true),
        });
      });
      return shouldBlock;
    },
    withResolver: true,
    // beforeunload страницы может сработать
    enableBeforeUnload: () => formIsDirty,
  });

  return (
    <>
      {/* вариант с отображением кастомного блок-компонента */}
      {status === "blocked" && (
        <div>
          <p>Are you sure you want to leave?</p>
          <button onClick={proceed}>Yes</button>
          <button onClick={reset}>No</button>
        </div>
      )}
    </>
  );

  // ...
}
```
