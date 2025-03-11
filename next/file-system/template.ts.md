оборачивает макет или страницу, в отличает от layout сбрасывают состояние при переходе

```tsx
export default function Template({ children }: { children: React.ReactNode }) {
  return <div>{children}</div>;
}
```
