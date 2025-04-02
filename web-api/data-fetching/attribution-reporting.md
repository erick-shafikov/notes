# API Attribution Reporting (-s)

позволяет измерять конверсию

- заголовок Attribution-Reporting-Eligible для разрешения регистрации
- взаимодействие через ссылку "а"
- предоставляет отчеты по событиям

```html
<!-- активация с помощью добавления в теги -->
<img
  src="advertising-image.png"
  attributionsrc="https://a.example/register-source"
/>
<script src="advertising-script.js" attributionsrc></script>
```

в fetch

```js
const attributionReporting = {
  eventSourceEligible: true,
  triggerEligible: false,
};

function triggerSourceInteraction() {
  fetch("https://shop.example/endpoint", {
    keepalive: true,
    attributionReporting,
  });
}

elem.addEventListener("click", triggerSourceInteraction);

// или window open

window.open(
  "https://ourshop.example",
  "_blank",
  `attributionsrc=${encodedUrlA},attributionsrc=${encodedUrlB}`
);
```
