# Хуки

## хук useForms

Позволяет отслеживать и управлять формой, в которую передан control

```js
const {
  register,
  unregister,
  formState,
  watch,
  handleSubmit,
  reset,
  resetField,
  setError,
  clearErrors,
  setValue,
  setFocus,
  getValues,
  getFieldState,
  trigger,
  control,
  Form,
} = useForms({
  /* Валидация полей до подтверждения формы
   'onChange' - на каждое изменение формы (бьет по производительности)
   'onBlur - при потери фокуса
   'onTouched' - после первой потери фокуса 
   'all' = onBlur + onChange
  */
  mode: "onChange" | "onBlur" | "onSubmit" | "onTouched" | "all",
  // повторная валидация после подтверждения формы
  reValidateMode: "onChange" | "onBlur" | "onSubmit",
  //значения по умолчанию далее FieldValues
  defaultValues: {
    __fieldName__: "__fieldName__",
    // ...
  },
  // может быть и асинхронной функицей
  defaultValues: async () => fetch("/api-endpoint"),
  // асинхронно подгружаемые значения
  values: FieldValues,
  // асинхронно подгружаемые значения ошибок
  errors: FieldErrors,
  resetOptions: {
    keepDirtyValues: true, // вводимые пользователем данные будут сохранены
    keepErrors: true, // ошибки ввода будут сохранены при обновлении значения
  },
  //может использоваться вторым аргументом для Yup, мутабельный
  context: object,
  //режим отображения ошибки firstError - только первая, all - все
  criteriaMode: firstError | all,
  // Наведет на первую ошибку
  shouldFocusError: (boolean = true),
  //отображение ошибки через мс
  delayError: number,
  // поведение при отключении поля от формы
  shouldUnregister: (boolean = false),
  // нативная валидация элементов
  shouldUseNativeValidation: (boolean = false),
  // для валидации
  resolver: Resolver({
    values: object, //содержит значения формы
    context: object, //мутабельный объект который может быть изменен при ререндере
    options: {
      criteriaMode: "string",
      fields: "object",
      names: "string[]",
    },
  }),
});
```

### const { register } = useForms()

Метод позволяет регистрировать поля формы

```js
const { register } = useForms()

const { onChange, onBlur, name, ref } = register('firstName');
// Вариант с передачей пропсов
<input onChange={onChange} onBlur={onBlur} name={name} ref={ref} />
// Вариант со spread
<input {...register('firstName')} />

// register("firstName") ===	{firstName: 'value'}
// register("name.firstName")	{name: { firstName: 'value' }}
// register("name.firstName.0")	{name: { firstName: [ 'value' ] }}

// варианты без и с текстом ошибки
// name 'test.0.firstName' - в случае fieldArray
<input {...register('firstName', {
  required: string | {value: boolean, message: string},
  maxLength: number | {value: boolean, message: string},
  minLength: number | {value: boolean, message: string},
  max: number | {value: boolean, message: string},
  min: number | {value: boolean, message: string},
  pattern: Regex | {value: Regex, message: string},
  // валидация поля
  validate: (fieldValue) => boolean | {
    validationFunc1:  (fieldValue) => boolean,
    validationFunc2:  (fieldValue) => boolean,
  },
  valueAsNumber: boolean,
  valueAsDate: boolean,
  // изменяет вводимое значение
  setValueAs: (value) => value,
  disabled: boolean,
  onChange: (e: SyntheticEvent) => void,
  onBlur: (e: SyntheticEvent) => void,
  value: unknown,
  // удаления из контекста
  shouldUnregister: booleanб
  deps: string | [string]
})} />
```

### const { unregister } = useForms()

Функция. которая позволяет открепить от контекста

### const { formState } = useForms()

Объект, который содержит информацию о форме

```js
const { formState } = useForms();

const formState: {
  isDirty: boolean,
  dirtyFields: {
    __fieldName__: boolean
    },
    touchedFields: {
    __fieldName__: boolean
    },
    defaultValues: FormValuesObject,
    isSubmitted: boolean,
    isSubmitSuccessful:  boolean,
    isSubmitting: boolean,
    isLoading: boolean,
    submitCount: number,
    isValid: boolean,
    isValidating: boolean,
    errors: {
      __fieldName__: ' fieldName error message'
    }
}
```

### const { watch } = useForms()

Функция которая позволяет отслеживать значение полей

```js
const { watch } = useForms();

const fieldValue = watch("inputName"); // fieldValue значение поля
const fieldValue = watch(["inputName1"]); // fieldValue - массив значений полей
const fieldValue = watch(); // вернет значения всех полей {[key:string]: unknown}
const fieldValue = watch((data, { name, type }) =>
  console.log(data, name, type)
); // вернет	функцию для отписки { unsubscribe: () => void }
```

### const { handleSubmit } = useForms()

Функция для подтверждения формы, выполниться только при успешной валидации

```js
const { handleSubmit } = useForms();

const onSubmit = async () => {
  // async request which may result error
  try {
    // await fetch()
  } catch (e) {
    // handle your error
  }
};

<form onSubmit={handleSubmit(onSubmit)} />;
```

### const { reset } = useForms()

Функция для сброса значений формы

```js
const { reset } = useForms();

reset({
  values: {},
  // сброс ошибок
  keepErrors: boolean,
  // сброс состояния
  keepDirty: boolean,
  // сброс состояния только неизмененных полей
  keepDirtyValues: boolean,
  // значения в форме будут не изменены
  keepValues: boolean,
  keepDefaultValues: boolean,
  keepIsSubmitted: boolean,
  keepTouched: boolean,
  keepIsValid: boolean,
  keepSubmitCount: boolean,
});
```

### const { resetField } = useForms()

Позволяет осуществить сброс конкретного поля

```js
// первый аргумент имя поля
const handleClick = () =>
  resetField("firstName", {
    keepError: boolean,
    keepDirty: boolean,
    keepTouched: boolean,
    defaultValue: "new value",
  });
```

### const { setError } = useForms()

позволяет поставить ошибку

```js
{ setError } = useForms();
setError('__inputName__', {
  error: {
    type: 'custom', //тип ошибки валидации
    message: 'custom message' //сообщение
    },
  config: {
    shouldFocus?: boolean
  }
});
```

### const { clearErrors } = useForms()

позволяет удалить ошибку

```js
const { clearErrors } = useForms();
clearErrors(); //уберет все ошибки
clearErrors("yourDetails.firstName"); //уберет ошибки конкретного поля
clearErrors(["yourDetails.lastName"]); //уберет ошибки из указанных полей
```

### const { setValue } = useForms()

Императивно установит значение в поле

```js
const { setValue } = useForms();

setValue("fieldName", "fieldValue", {
  shouldValidate: boolean,
  shouldValidate: boolean,,
  shouldTouch: boolean,
});

// для вложенных полей
setValue('yourDetails.firstName', 'value');
setValue('nestedValue', { test: 'updatedData' } );
```

### const { setFocus } = useForms()

Позволяет установить фокус

```js
const { setFocus } = useForms();

setFocus("__fieldName__", {
  shouldSelect: boolean, //выбрать контент внутри поля
});
```

### const { getValues } = useForms()

функция, позволяющая получить значение поля

```js
const { getValues } = useForms();

getValue(); //вернет значения всех полей
getValue("__fieldValue_"); //вернет значение конкретного поля
getValue(["__fieldValue_", "__fieldValue_"]); //вернет значение конкретных полей
```

### const { getFieldState } = useForms()

функция, при вызове которой можно получить состояние поля

```js
const { getFieldState } = useForms();

const {
  isDirty: boolean,
  isTouched: boolean,
  invalid: boolean,
  error: Error,
} = getFieldState("__fieldName__", { formState });
```

### const { trigger } = useForms()

Функция позволяет инициализировать валидацию

```js
const { trigger } = useForms();

trigger(); //на всех полях
trigger("__fieldName__"); //на конкретном
trigger(["__fieldName__", "__fieldName__"]); //на конкретных
```

### const { control } = useForms()

объект содержит методы для регистрации поля

```js
import { TextField } from "@material-ui/core";

//...
const { control } = useForms();

<Controller
  as={TextField}
  name="firstName"
  control={control}
  defaultValue=""
/>;

//...
```

## useController

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

## useFormContext

для передачи контекст формы

```js
const methods = useForm()

<FormProvider {...methods} /> // all the useForm return props

const methods = useFormContext() // useForm для вложенных компонентов
```

## useWatch

хук подобный watch, но с лучшей производительностью

```js
const {} = useWatch({
  name: "__fieldName__" | ["__fieldName__", "__fieldName__"],
  control: control,
  defaultValue: "__fieldName__",
  disabled: boolean, //false
  exact: boolean, //false
});
```

## useFromState

позволяет получить состояние формы

```js
import * as React from "react";
import { useForm, useFormState } from "react-hook-form";

export default function App() {
  const { register, handleSubmit, control } = useForm({
    defaultValues: {
      firstName: "firstName",
    },
  });
  const onSubmit = (data) => console.log(data);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register("firstName")} placeholder="First Name" />
      {/* передаем control*/}
      <Child control={control} />

      <input type="submit" />
    </form>
  );
}
// получаем control
function Child({ control }) {
  const { dirtyFields } = useFormState({
    control,
  });

  return dirtyFields.firstName ? <p>Field is dirty.</p> : null;
}
```

## useFieldArray

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

  React.useEffect(() => {
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

# Компоненты

## Controller

Компонент для подключения инпутов у react hook forms

```js
import ReactDatePicker from "react-datepicker"
import { TextField } from "@material-ui/core"
import { useForm, Controller } from "react-hook-form"

type FormValues = {
  ReactDatePicker: string
}

function App() {
  const { handleSubmit, control } = useForm<FormValues>()

  return (
    <form onSubmit={handleSubmit((data) => console.log(data))}>
      <Controller
        control={control}
        name="ReactDatePicker"
        render={({ field: { onChange, onBlur, value, ref } }) => (
          <ReactDatePicker
            onChange={onChange} // send value to hook form
            onBlur={onBlur} // notify when input is touched/blur
            selected={value}
          />
        )}
      />

      <input type="submit" />
    </form>
  )
}
```

## Form

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

# React Native

## React Native и Form компонент

```js
// react native
import { useForm, Form } from "react-hook-form";
function App() {
  const {
    control,
    register,
    formState: { isSubmitSuccessful, errors },
  } = useForm();

  return (
    <Form
      action="/api"
      control={control}
      render={({ submit }) => {
        <View>
          {isSubmitSuccessful && <Text>Form submit successful.</Text>}

          {errors?.root?.server && <Text>Form submit failed.</Text>}
          <Button onPress={() => submit()} />
        </View>;
      }}
    />
  );
}
```

## React Native и Controller компонент

```js
import { Text, View, TextInput, Button, Alert } from "react-native";
import { useForm, Controller } from "react-hook-form";

export default function App() {
  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues: {
      firstName: "",
      lastName: "",
    },
  });
  const onSubmit = (data) => console.log(data);

  return (
    <View>
      <Controller
        control={control}
        rules={{
          required: true,
        }}
        render={({ field: { onChange, onBlur, value } }) => (
          <TextInput
            placeholder="First name"
            onBlur={onBlur}
            onChangeText={onChange}
            value={value}
          />
        )}
        name="firstName"
      />
      {errors.firstName && <Text>This is required.</Text>}

      <Controller
        control={control}
        rules={{
          maxLength: 100,
        }}
        render={({ field: { onChange, onBlur, value } }) => (
          <TextInput
            placeholder="Last name"
            onBlur={onBlur}
            onChangeText={onChange}
            value={value}
          />
        )}
        name="lastName"
      />

      <Button title="Submit" onPress={handleSubmit(onSubmit)} />
    </View>
  );
}
```
