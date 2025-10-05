# Form

```js
<Form
  control={control}
  children={}
  // для headless компонентов
  render={({ submit }) => <View/>}
  onSubmit={() => {}} // Функция вызываемая перед запросом
  onSuccess={() => {}} // при успешно валидации
  onError={() => {}} // при валидации с ошибками
  // для заголовков
  headers={{ accessToken:  'xxx', 'Content-Type':  'application/json'  }}
  action="/api"
  method="post" // default to post
  validateStatus={(status) => status >= 200} // validate status code
/>
```
