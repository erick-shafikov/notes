# useForms

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

# Объект возврата

## register

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

## unregister

Функция. которая позволяет открепить от контекста

## formState

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

## watch

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

## handleSubmit

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

## reset

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

## resetField

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

## setError

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

## clearErrors

позволяет удалить ошибку

```js
const { clearErrors } = useForms();
clearErrors(); //уберет все ошибки
clearErrors("yourDetails.firstName"); //уберет ошибки конкретного поля
clearErrors(["yourDetails.lastName"]); //уберет ошибки из указанных полей
```

## setValue

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

## setFocus

Позволяет установить фокус

```js
const { setFocus } = useForms();

setFocus("__fieldName__", {
  shouldSelect: boolean, //выбрать контент внутри поля
});
```

## getValues

функция, позволяющая получить значение поля

```js
const { getValues } = useForms();

getValue(); //вернет значения всех полей
getValue("__fieldValue_"); //вернет значение конкретного поля
getValue(["__fieldValue_", "__fieldValue_"]); //вернет значение конкретных полей
```

## getFieldState

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

## trigger

Функция позволяет инициализировать валидацию

```js
const { trigger } = useForms();

trigger(); //на всех полях
trigger("__fieldName__"); //на конкретном
trigger(["__fieldName__", "__fieldName__"]); //на конкретных
```

## control

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
