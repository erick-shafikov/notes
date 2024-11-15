<!-- js ----------------------------------------------------------------------------------------------------------------------------------->

# js

Убирать дефолтные и тестовые значения

## js. code formatting

- enter перед return
- относительные импорты

- [ошибка при использовании || и ?? с потенциально числовыми значениями](./js/js-core/logical-operators.md#bp-операторы--и--с-потенциально-числовыми-значениями)

- [Ошибка в обращении к полям после проверки на undefined](./js/js-core/logical-operators.md#bp-ошибка-в-обращении-к-полям-после-проверки-на-undefined)

- [переиспользование](./js/js-core/js-data-types/object.md#bp-переиспользование-переменных)

<!-- react ----------------------------------------------------------------------------------------------------------------------------------->

# react

## react. naming

- Пропсы называются OnChangePage, функции-обработчики событий через handleChangePage
- при формировании функций handleCloseModal => handleModalClose
- CreateSmzClientAction -> SmzClientCreateAction
- стиль наименования коллбеков - handle[сущность][действие]

- из хуков

```js
return {
  is__Entity__Fetching: isFetching,
  is__Entity__Previous: isPreviousData,
  __Entity__Total: total,
  __Entity__: items,
};
```

## react. стилизация

- больше трех в prop - верстка в styled

- [цвет svg](./html/svg.md#currentColor)

## react. логика

- если какая-либо переменная использует из внешнего контекста внутри useMemo или useCallback, то нужно добавить в зависимость
- не ставить лишних async в ten stack query
