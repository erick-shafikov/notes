## useFormContext

для передачи контекст формы

```js
const methods = useForm()

<FormProvider {...methods} /> // all the useForm return props

const methods = useFormContext() // useForm для вложенных компонентов
```
