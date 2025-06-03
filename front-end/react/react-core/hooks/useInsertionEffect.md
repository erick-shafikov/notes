## useInsertionEffect

useInsertionEffect позволяет вставлять элементы в DOM до срабатывания каких-либо эффектов макета.

Пример, где в тег style добавляется дополнительный стиль

```jsx
let isInserted = new Set();
function useCSS(rule) {
  useInsertionEffect(() => {
    // As explained earlier, we don't recommend runtime injection of <style> tags.
    // But if you have to do it, then it's important to do in useInsertionEffect.
    if (!isInserted.has(rule)) {
      isInserted.add(rule);
      document.head.appendChild(getStyleForRule(rule));
    }
  });
  return rule;
}

function Button() {
  const className = useCSS("...");
  return <div className={className} />;
}
```
