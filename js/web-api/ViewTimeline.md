ViewTimeline позволяет создать анимации на основе прокрутки scroll контейнера

```js
document.addEventListener("DOMContentLoaded", () => {
  const container1 = document.querySelector(".c-1");
  const container2 = document.querySelector(".c-2");
  const progress = document.querySelector(".c-1 .progress .progress-inner");

  //создание контекста
  const viewTimeline = new ViewTimeline({
    subject: progress,
    axis: "block",
    // inset: ["auto", CSS.px("100")],
    // rangeStart: "cover 30%",
    rangeStart: {
      offset: CSS.percent("30"),
    },
  });

  //использование
  progress.animate(
    [
      {
        width: 0,
      },
      {
        width: "100%",
      },
    ],
    { fill: "both", timeline: viewTimeline }
  );
});
```
