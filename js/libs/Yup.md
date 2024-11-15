```ts
const FormSchema = Yup.object({
  // простая валидация --------------------------------------------------
  // валидация текстового поля
  stringField: Yup.string().required("Обязательное поле"),
  email: Yup.string()
    .email("Неверный формат электронной почты")
    .required("Обязательное поле"),
  phone: Yup.string()
    .matches(/^((\+7|7|8)+([0-9]){10})$/, {
      message: "Указан недействительный номер",
    })
  // сложная валидация---------------------------------------------------
  // не должны совпадать
  noEqualField: Yup.string()
    .notOneOf(
      [Yup.ref("PossibleEqualField")],
      "Старый и новый пароли не должны совпадать"
    )
    .required("Обязательное поле"),
  //должны совпадать
  equalField: Yup.string()
    .oneOf([Yup.ref("otherEqualField")], "Пароли не совпадают")
    .required("Обязательное поле"),
  // тест строкового значения, как числового
  amount: Yup.string()
    .min(0)
    .test('more than 1kk', 'Не более 1 млн.', (value = 0) => Number(value) <= 1000000)
    .required('Обязательное поле'),
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
const FormSchema = Yup.object().shape({
  __dependedField__: Yup.string().when(
    ["__field1__", "__field2__"], // (2 аргумента)
    __validationUtil__YupBuilder
  ),
});
```

!!!TODO добавить к схеме выше

```js
const Schema = Yup.object({
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
  // валидация электронной почты
  email: Yup.string()
    .trim()
    .email("Неверный формат электронной почты")
    .required("Обязательное поле"),

  // --------------------------------------------------------------------
  //зависимые поля
  controlFieldOne: Yup.string(),
  controlFieldTwo: Yup.string(),
  dependedField: Yup.object()
    //если одно поле в качестве управления
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
    })
    // два и более поля
    .when(["controlFieldOne", "controlFieldTwo"], {
      is: (controlFieldOne, controlFieldTwo) => {},
      then: (schema) => schema.shape({}),
    }),
  //валидация числовых значений через вложенные схемы в обход циклических зависимостей
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
  // валидация объектов
  // валидация массивов
  expiredAt: Yup.array().when("isIndefinitely", {
    is: false,
    then: (schema) => schema.of(Yup.date()).required("Обязательное поле"),
  }),
});
```
