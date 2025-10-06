<!-- calc-size()---------------------------------------------------------------------------------------------------------------------------->

# calc-size() (-ff -safari)

Позволяет вычислять размеры для ключевых слов auto, fit-content, max-content, content, max-content.

```scss
.calc-size {
  // получение значений calc-size(), ничего не делать со значениями
  height: calc-size(auto, size);
  height: calc-size(fit-content, size);
  // применять изменения к измеримому объекту
  height: calc-size(min-content, size + 100px);
  height: calc-size(fit-content, size / 2);
  // с функциями
  height: calc-size(auto, round(up, size, 50px));
}
```

```scss
section {
  height: calc-size(calc-size(max-content, size), size + 2rem);
}

//тоже самое что и
:root {
  --intrinsic-size: calc-size(max-content, size);
}

section {
  height: calc-size(var(--intrinsic-size), size + 2rem);
}
```

<!-- calc() ---------------------------------------------------------------------------------------------------------------------------------->

# calc()

Применение различных операций

```scss
 {
  // 110px
  width: calc(10px + 100px);
  // 10em
  width: calc(2em * 5);
  // в зависимости от ширины
  width: calc(100% - 32px);
  --predefined-width: 100%;
  /* Output width: Depends on the container's width */
  width: calc(var(--predefined-width) - calc(16px * 2));
}
```

- !!! Типы должны сходится

<!-- fit-content()  ------------------------------------------------------------------------------------------------------------------------>

# fit-content()

Позволяет взять размер меньший из максимального и максимального между минимальным и заданным

fit-content(argument) = min(maximum size, max(minimum size, argument)).
