# Валидация простых полей

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

## тест строкового значения, как числового

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

## Валидация инлайн даты

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

# Зависимые проверки

## два поля не должны совпадать

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

## два поля должны совпадать

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

## зависимые поля (одно)

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

## зависимые поля (два и более)

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

# BP. разделение зависимой валидации на отдельные утилиты

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

# Валидация вложенных объектов

```js
const Schema = Yup.schema({
  client: Yup.object({
    fullName: Yup.string(),
    phone: Yup.string(),
    passport: Yup.string(),
  }),
});
```

# Валидация массива

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
