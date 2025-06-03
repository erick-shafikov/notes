## useRouter

позволяет осуществлять роутинг

```js
const router = useRouter({ scroll: Boolean });

router.push(pathname); //добавляет в history
router.replace(pathname); //не добавляет в history,
router.refresh(); //делает повторный запрос на сервер, без потери состояния если это CC,
router.prefetch(path); //префетч определенного роута, forward: func(), back()
```
