<!-- Profiler ------------------------------------------------------------------------------------------------------------------------------>

# Profiler

Для отладки компонентов

```js
function onRender(
  id,
  phase,
  actualDuration,
  baseDuration,
  startTime,
  commitTime
) {
  // Aggregate or log render timings...
}
<Profiler id="App" onRender={onRender}>
  <App />
</Profiler>;
```
