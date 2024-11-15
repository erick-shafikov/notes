ScrollTimeline дял создания анимации использующих scroll

```js
document.addEventListener("DOMContentLoaded", () => {
  const container1 = document.querySelector(".c-1");
  const container2 = document.querySelector(".c-2");

  // создаем timeline контекст
  const timeline = new ScrollTimeline({
    // привязываемся
    source: container2,
    axis: "block",
  });

  container1.animate(
    [
      {
        backgroundColor: "salmon",
      },
    ],
    {
      fill: "both",
      // добавляем
      timeline,
    }
  );
});
```
