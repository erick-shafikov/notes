# React.ComponentProps

```tsx
function Button(props: React.ComponentProps<"button">) {
  return <button {...props} />;
}
```

# JSX.Element

JSX.Element - Один результат JSX-выражения (то, что возвращает компонент)

```jsx
<div>Hello</div>
```

# ReactNode

Любое значение, которое React может отрендерить null, string, number, JSX.Element, ReactFragment, ReactPortal, и т.д.
