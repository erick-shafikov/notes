## userAgent

функция, которая принимает в параметры запрос, возвращает

```js
const userAgentObject = userAgent(request);
const userAgentObject = {
  isBot: bool,
  browser: { name, version },
  device: { model, type: "console" | "tablet" | "smarttv" },
  engine: String,
  os: String,
  cpu: String,
};
```
