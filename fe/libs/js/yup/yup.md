# test()

## test() вариант 1

```ts
type test = (
  name: string, //обязательное поле
  message: string | function | any, //текст ошибки ${param} - синтаксис
  test: () => true | ValidationError
) => Schema;
```

```js
let jimmySchema = string().test(
  "is-jimmy",
  "${path} is not Jimmy",
  (value, context) => value === "jimmy"
);

// or make it async by returning a promise
let asyncJimmySchema = string()
  .label("First name")
  .test(
    "is-jimmy",
    ({ label }) => `${label} is not Jimmy`, // a message can also be a function
    async (value, testContext) =>
      (await fetch("/is-jimmy/" + value)).responseText === "true"
  );

await schema.isValid("jimmy"); // => true
await schema.isValid("john"); // => false
```

где контекст

```ts
type Context = {
  path: string; // строковый путь текущей проверки
  schema: Schema; // разрешенный объект схемы, с которым выполняется тест.
  options: object; // объект, для которого был вызван метод validate() или isValid()
  parent: object; //в случае вложенной схемы это значение родительского объекта
  originalValue: any; //исходное значение, которое проверяется
  createError: (Object: {
    path: String;
    message: String;
    params: Object;
  }) => any; //создать и вернуть ошибку проверки. Полезно для динамической установки path, params, или, что более вероятно, ошибки message. Если какой-либо из параметров пропущен, будет использован текущий путь или сообщение по умолчанию.
};
```

## test() вариант 2

```ts
type test = (options: {
  name: string; // имя теста
  test: (value: any) => boolean; // тест со значением
  message: string; // текст ошибки
  params?: object; //параметры
  exclusive: boolean; // false;
}) => Schema;
```

```ts
let max = 64;
let schema = yup.string().test({
  name: "max",
  exclusive: true,
  params: { max },
  message: "${path} must be less than ${max} characters",
  test: (value) => value == null || value.length <= max,
});
```

# when

```ts
type when = (keys: string | string[], builder: object | (values: any[], schema) => Schema) => Schema
```

Вариант с возвратом схемы в виде объекта, ключи is, then, otherwise

```ts
let schema = object({
  isBig: boolean(),
  count: number()
    .when("isBig", {
      is: true, // alternatively: (val) => val == true
      then: (schema) => schema.min(5), //(schema: Schema) => Schema
      otherwise: (schema) => schema.min(0),
    })
    .when("$other", ([other], schema) =>
      other === 4 ? schema.max(6) : schema
    ),
});

await schema.validate(value, { context: { other: 4 } });
```

Для двх и более зависимых полей

```ts
let schema = object({
  isSpecial: boolean(),
  isBig: boolean(),
  count: number().when(["isBig", "isSpecial"], {
    is: true, // alternatively: (isBig, isSpecial) => isBig && isSpecial
    then: (schema) => schema.min(5),
    otherwise: (schema) => schema.min(0),
  }),
});

await schema.validate({
  isBig: true,
  isSpecial: true,
  count: 10,
});
```

Функция сравнения в виде массива

```ts
let schema = yup.object({
  isBig: yup.boolean(),
  count: yup.number().when("isBig", ([isBig], schema) => {
    return isBig ? schema.min(5) : schema.min(0);
  }),
});

await schema.validate({ isBig: false, count: 4 });
```

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs:

## BP. Валидация простых полей

[регулярные выражения](../js-core/regexp/regex-examples.md)

```ts
const Schema = Yup.object({
  // простая валидация --------------------------------------------------
  // валидация текстового поля
  stringField: Yup.string().required("Обязательное поле"),
  email: Yup.string()
    .trim()
    .email("Неверный формат электронной почты")
    .required("Обязательное поле"),
  //валидация строк
  simpleStringValue: Yup.string()
    .trim()
    .required("Обязательное поле")
    .trim()
    .min(8, "Требуется минимум 8 символов")
    .max(36, "Не более 36 символов"),
  // валидация номера телефона
  phone: Yup.string()
    .required("Обязательное поле")
    .matches(/^((\+7|7|8)+([0-9]){10})$/, {
      message: "Указан недействительный номер",
    }),
  // валидация электронной ИНН
  inn: Yup.string().test({
    name: "inn-code-test",
    test: (innToValidate) => !innToValidate || validateInn(innToValidate),
    message: "Проверьте, пожалуйста ИНН",
  }),
});
```

### тест строкового значения, как числового

```js
const Schema = Yup.object({
  amount: Yup.string()
    .min(0)
    .test(
      "more than 1kk",
      "Не более 1 млн.",
      (value = 0) => 0 >= Number(value) && Number(value) <= 1000000
    )
    .required("Обязательное поле"),
});
```

### Валидация инлайн даты

Если есть текстовый ввод значения с маской

```js
const Schema = Yup.object({
  issueDate: Yup.date()
    .nullable()
    .notRequired()
    .transform((_, originalValue) => {
      if (originalValue) {
        const transformedDate = transformDate(
          originalValue,
          "DD/MM/YYYY",
          "MM/DD/YYYY"
        );

        return isNaN(transformedDate) ? new Date() : transformedDate;
      }
    })
    .typeError("Неверный формат даты")
    // или мин
    .max(formatDateTime(new Date(), "dd.MM.yyyy"), "Введите верную дату"),
});
```

## BP. Зависимые проверки

### два поля не должны совпадать

```js
const Schema = Yup.object({
  noEqualField: Yup.string()
    .notOneOf(
      [Yup.ref("PossibleEqualField")],
      "Старый и новый пароли не должны совпадать"
    )
    .required("Обязательное поле"),
});
```

### два поля должны совпадать

```js
const Schema = Yup.object({
  equalField: Yup.string()
    .oneOf([Yup.ref("otherEqualField")], "Пароли не совпадают")
    .required("Обязательное поле"),
});
```

```js
const Schema = Yup.object({
  amountTo: Yup.string().when("amountFrom", (amountFromValue, schema) => {
    if (amountFromValue) {
      return schema.test(
        "amount-to-boundary",
        "Не менее чем сумма(от)",
        (amountToValue) => {
          return amountToValue
            ? Number(amountToValue) >= Number(amountFromValue)
            : true;
        }
      );
    }

    return schema;
  }),
});
```

### зависимые поля (одно)

```js
const Schema = Yup.object({
  controlFieldOne: Yup.string(),
  controlFieldTwo: Yup.string(),
  dependedField: Yup.object()
    //если одно поле в качестве управления для все формы
    .when("controlFieldOne", {
      is: (controlFieldOne) => {}, //boolean
      then: (schema) =>
        schema.shape({
          phone: Yup.string()
            .required("Обязательное поле")
            .matches(/^((\+7|7|8)+([0-9]){10})$/, {
              message: "Указан недействительный номер",
            }),
        }), //схема на возврат
    }),
});
```

### зависимые поля (два и более)

```js
const Schema = Yup.object({
  controlFieldOne: Yup.string(),
  controlFieldTwo: Yup.string(),
  dependedField: Yup.object()
    // два и более поля
    .when(["controlFieldOne", "controlFieldTwo"], {
      is: (controlFieldOne, controlFieldTwo) => {},
      then: (schema) => schema.shape({}),
    }),
  // валидация объектов
  // валидация массивов
  expiredAt: Yup.array().when("isIndefinitely", {
    is: false,
    then: (schema) => schema.of(Yup.date()).required("Обязательное поле"),
  }),
});
```

## BP. разделение зависимой валидации на отдельные утилиты

```ts
// утилита для валидации (2 аргумента)
const __validationUtil__YupBuilder = ([__field1__, __field2__]: any) => {
  if (
    // проверка
  ) {
    return Yup.string() //первая схема валидации

  } else {
    // альтернативная схема
    return Yup.string();
  }
};

// общая схема валидации формы
const Schema = Yup.object().shape({
  __dependedField__: Yup.string().when(
    ["__field1__", "__field2__"], // (2 аргумента)
    __validationUtil__YupBuilder
  ),
});
```

## BP. Валидация вложенных объектов

```js
const Schema = Yup.schema({
  client: Yup.object({
    fullName: Yup.string(),
    phone: Yup.string(),
    passport: Yup.string(),
  }),
});
```

## BP. Валидация массива

```js
const Schema = Yup.object({
  // если есть вложенные зависимые поля
  products: Yup.array().when("measurementUnit", (measurementUnit, schema) =>
    schema
      .of(
        Yup.object().shape({
          name: Yup.string().trim(),
          weightTare: Yup.string()
            .trim()
            // вложенное внутренние поле
            .when("weightBrutto", (weightBrutto) => {
              const parsedWeightBrutto = Number(weightBrutto);

              if (parsedWeightBrutto) {
                return Yup.number()
                  .max(
                    parsedWeightBrutto,
                    "Значение должно быть не больше веса брутто"
                  )
                  .required("Обязательное поле");
              }

              return Yup.string().trim();
            }),
          //ссылка на внешнее значение
          weightNetto: Yup.string().when("name", (name) => {
            if (name) {
              return measurementUnit === MEASUREMENT_UNIT_PIECE
                ? Yup.number()
                    .integer(
                      'При выборе единцы измерения "шт", данное поле должно быть целым числом'
                    )
                    .required("Обязательное поле")
                : Yup.string().trim().required("Обязательное поле");
            }
          }),
          type: Yup.string().when("name", (name) => {
            return name
              ? Yup.string().trim().required("Обязательное поле")
              : Yup.string().trim();
          }),
        })
      )
      .min(1, "Обязательно поле")
  ),
});
```
