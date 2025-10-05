# useController

Кастомный хук для активации Controller компонента. для создания переиспользуемого контролируемого input

```js
const {
  onChange,
  onBlur,
  value,
  disabled,
  name,
  ref,
  invalid,
  isTouched,
  isDirty,
  error,
} = useController({
  control: control, //объект контроля
  defaultValue: "__someDefaultValue__",
  rules: {},
  shouldUnregister: boolean,
  disabled: boolean,
});
```
