## useParams

получить params

## usePathname

получить путь

## useRouter

позволяет осуществлять роутинг

```js
const router = useRouter({ scroll: Boolean });

router.push(pathname); //добавляет в history
router.replace(pathname); //не добавляет в history,
router.refresh(); //делает повторный запрос на сервер, без потери состояния если это CC,
router.prefetch(path); //префетч определенного роута, forward: func(), back()
```

## useSearchParams

получить SearchParams

## useSelectedLayoutSegment

хук клиентских компонентов, позволяющий прочитать активный сегмент на уровень ниже, возвращает строку или null. Использование – стилизация активных link

## useSelectedLayoutSegments

хук, который позволяет прочитать активный роут выше на все уровни и вернет массив из строк

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
