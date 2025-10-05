## FormProvider

позволяет обернуть в контекст

```js
import React from "react";

import { useForm, FormProvider, useFormContext } from "react-hook-form";

export default function App() {
  // достаем методы для работы с формой
  const methods = useForm();

  const onSubmit = (data) => console.log(data);

  return (
    // передаем в Provider
    <FormProvider {...methods}>
      // pass all methods into the context
      <form onSubmit={methods.handleSubmit(onSubmit)}>
        <NestedInput />
        <input type="submit" />
      </form>
    </FormProvider>
  );
}

function NestedInput() {
  // можем использовать во вложенных
  const { register } = useFormContext(); // retrieve all hook methods

  return <input {...register("test")} />;
}
```
