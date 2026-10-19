<!-- Profiler ------------------------------------------------------------------------------------------------------------------------------>

# Profiler

Для отладки компонентов

```tsx
function onRender(
  id: string,
  phase: "mount" | "update",
  actualDuration: number,
  baseDuration: number,
  startTime: number,
  commitTime: number
): void {
  // Aggregate or log render timings...
}
<Profiler id="App" onRender={onRender}>
  <App />
</Profiler>;
```
