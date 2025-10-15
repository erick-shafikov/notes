# open

```ts
function open(
  url?: string, //нужный ресурс
  target?: string, // "_blank" | "_self"  | "_parent" | "_top," | либо строка которая будет являться названием окна
  windowFeatures?: string // строка, которая содержит функции окна формата name=value
): WindowProxy | null;
```

Параметры строки windowFeatures:

- attributionsrc
- popup
- width или innerWidth
- height или innerHeight
- left или screenX
- top или screenY
- noopener
- noreferrer

Возвращает WindowProxy нового окна, если успешно отурыто окно, null если нет. Если открытый контекст не из того же источника, то скрипт не сможет взаимодействовать

```js
const windowFeatures = "left=100,top=100,width=320,height=320";
const handle = window.open(
  "https://www.mozilla.org/",
  "mozillaWindow",
  windowFeatures
);
if (!handle) {
  // The window wasn't allowed to open
  // This is likely caused by built-in popup blockers.
  // …
}
```
