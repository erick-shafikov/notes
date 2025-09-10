```tsx
import { Block } from "@tanstack/react-router";

function MyComponent() {
  const [formIsDirty, setFormIsDirty] = useState(false);

  return (
    <Block
      shouldBlockFn={() => {
        if (!formIsDirty) return false;

        const shouldLeave = confirm("Are you sure you want to leave?");
        return !shouldLeave;
      }}
      enableBeforeUnload={formIsDirty}
    />
  );

  // OR

  return (
    <Block
      shouldBlockFn={() => formIsDirty}
      enableBeforeUnload={formIsDirty}
      withResolver
    >
      {({ status, proceed, reset }) => (
        <>
          {/* ... */}
          {status === "blocked" && (
            <div>
              <p>Are you sure you want to leave?</p>
              <button onClick={proceed}>Yes</button>
              <button onClick={reset}>No</button>
            </div>
          )}
        </>
      )}
    </Block>
  );
}
```
