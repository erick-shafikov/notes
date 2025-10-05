# useFieldArray

Хук позволяет работать с динамическими формами

```js
const {
  fields,
  append,
  prepend,
  insert,
  swap,
  update,
  replace,
  remove
} = useFieldArray({
  name: "__someArrayName__",
  control: control, //если нужно передать контекст какой-либо формы
  shouldUnregister: boolean, //будет снят с регистрации после анмаунта
  rules: object, //Объект с правилами валидации
});

function FieldArray() {
  const { control, register } = useForm();
  const { fields, append, prepend, remove, swap, move, insert } = useFieldArray({
    control, // control props comes from useForm (optional: if you are using FormProvider)
    name: "test", // unique name for your Field Array
  });

 useEffect(() => {
  remove(0);
}, [remove])

onClick={() => {
  append({ test: 'test' });
}}

  return (
    {fields.map((field, index) => (
      <input
        key={field.id} // important to include key with field's id
        {...register(`test.${index}.value`)}
      />
    ))}
  );
}
```

для ts

```ts
<input key={field.id} {...register(`test.${index}.test` as const)} />;

const { fields } = useFieldArray({
  name: `test.${index}.keyValue` as "test.0.keyValue",
});
```
