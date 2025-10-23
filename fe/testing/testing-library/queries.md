# queries

Позволяют найти элемент, но ведут себя по-разному при результате (0 | 1 | >1 совпадениях):

- getBy... - ошибка | элемент | ошибка
- queryBy... - null | элемент | ошибка
- findBy... - ошибка | элемент | ошибка, вернет промис
- getAllBy... - ошибка | массив | массив
- queryAllBy... - пустой массив | массив | массив
- findAllBy... ошибка | массив | массив, вернет промис

Приоритеты:

- запросы для всех категорий:
- - getByRole
- - getByLabelText
- - getByPlaceholderText
- - getByText
- - getByDisplayValue
- семантические:
- - getByAltText
- - getByTitle
- с тестовыми id
- - getByTestId

## findBy запросы

для асинхронных операций - появление на экране, промисы итд

```js
//без промиса
const button = screen.getByRole("button", { name: "Click Me" });
fireEvent.click(button);
//с промисом
await screen.findByText("Clicked once");
fireEvent.click(button);
await screen.findByText("Clicked twice");
```

# screen

document.body === screen

использование:

```js
import { screen, getByLabelText } from "@testing-library/dom";

// если использовать скрин, то элемент-контейнер всего приложения ненужен:
const inputNode1 = screen.getByLabelText("Username");

// если не использовать скрин, то элемент-контейнер всего приложения нужен:
const container = document.querySelector("#app");
const inputNode2 = getByLabelText(container, "Username");
```

допускается но тогда нужно передавать параметр container

```js
const { container } = render(<MyComponent />);
const foo = container.querySelector('[data-foo="bar"]');
```

<!-- TextMatch -->

# TextMatch

Все запросы принимаю этот аргумент

```js
// Может быть в виде строки:
screen.getByText("Hello World"); // full string match
screen.getByText("llo Worl", { exact: false }); // substring match
screen.getByText("hello world", { exact: false }); // ignore case

// regex:
screen.getByText(/World/); // substring match
screen.getByText(/world/i); // substring match, ignore case
screen.getByText(/^hello world$/i); // full string match, ignore case
screen.getByText(/Hello W?oRlD/i); // substring match, ignore case, searches for "hello world" or "hello orld"

// функции:
screen.getByText((content, element) => content.startsWith("Hello"));
screen.getByText((content, element) => {
  return (
    element.tagName.toLowerCase() === "span" && content.startsWith("Hello")
  );
});
```

<!-- Precision -->

# Precision

точность - { exact:true | false }
normalizer - переопределяет

```js
screen.getByText("text", {
  normalizer: getDefaultNormalizer({ trim: false }),
});
```

<!-- ByLabelText -->

# ByLabelText

Ищет по label, атрибутам for

getByLabelText, queryByLabelText, getAllByLabelText, queryAllByLabelText, findByLabelText, findAllByLabelText

```ts
function getByLabelText(
  // If you're using `screen`, then skip the container argument:
  container: HTMLElement,
  text: TextMatch,
  options?: {
    selector?: string = "*"; //тег
    exact?: boolean = true;
    normalizer?: NormalizerFn;
  }
): HTMLElement;
```

- [TextMatch](./queries.md#textmatch)

Варианты распознавания

```jsx
// for + id
<label for="username-input">Username</label>
<input id="username-input" />

// aria-labelledby + id
<label id="username-label">Username</label>
<input aria-labelledby="username-label" />

// вложенный
<label>Username <input /></label>

// вложенный с текстовым элементом
<label>
  <span>Username</span>
  <input />
</label>

// aria-label
<input aria-label="Username" />
```

Применение

```js
//native
import { screen } from "@testing-library/dom";

const inputNode = screen.getByLabelText("Username");
//react
import { render, screen } from "@testing-library/react";

render(<Login />);

const inputNode = screen.getByLabelText("Username");
```

<!-- ByRole -->

# ByRole

поиск по aria-атрибутам

getByRole, queryByRole, getAllByRole, queryAllByRole, findByRole, findAllByRole

```ts
function getByRole(
  // если используется scree, то необязательный параметр
  container: HTMLElement,
  role: string,
  options?: {
    // aria-атрибуты
    hidden?: boolean = false;
    name?: TextMatch;
    description?: TextMatch;
    selected?: boolean;
    busy?: boolean;
    checked?: boolean;
    pressed?: boolean;
    suggest?: boolean;
    current?: boolean | string;
    expanded?: boolean;
    queryFallbacks?: boolean;
    level?: number;
    value?: {
      min?: number;
      max?: number;
      now?: number;
      text?: TextMatch;
    };
  }
): HTMLElement;
```

<!-- ByPlaceholderText -->

# ByPlaceholderText

ищет по placeholder-ам

getByPlaceholderText, queryByPlaceholderText, getAllByPlaceholderText, queryAllByPlaceholderText, findByPlaceholderText, findAllByPlaceholderText

```ts
function getByPlaceholderText(
  container: HTMLElement, //ненужен если используется screen
  text: TextMatch,
  options?: {
    exact?: boolean = true;
    normalizer?: NormalizerFn;
  }
): HTMLElement;
```

<!-- ByText -->

# ByText

getByText, queryByText, getAllByText, queryAllByText, findByText, findAllByText

```ts
function getByText(
  container: HTMLElement, //...
  text: TextMatch,
  options?: {
    selector?: string = "*";
    exact?: boolean = true;
    ignore?: string | boolean = "script, style";
    normalizer?: NormalizerFn;
  }
): HTMLElement;
```

<!-- ByDisplayValue -->

# ByDisplayValue

вернет input, textarea, select если элемент имеет такое значение

getByDisplayValue, queryByDisplayValue, getAllByDisplayValue, queryAllByDisplayValue, findByDisplayValue, findAllByDisplayValue

<!-- ByAltText -->

# ByAltText

для img

```ts
function getByAltText(
  container: HTMLElement, //
  text: TextMatch,
  options?: {
    exact?: boolean = true;
    normalizer?: NormalizerFn;
  }
): HTMLElement;
```

<!-- ByTitle -->

# ByTitle

для svg элементов title

getByTitle, queryByTitle, getAllByTitle, queryAllByTitle, findByTitle, findAllByTitle

```ts
getByTitle(
  // If you're using `screen`, then skip the container argument:
  container: HTMLElement,
  title: TextMatch,
  options?: {
    exact?: boolean = true,
    normalizer?: NormalizerFn,
  }): HTMLElement
```

# ByTestId

для элементов с data-testid

getByTestId, queryByTestId, getAllByTestId, queryAllByTestId, findByTestId, findAllByTestId

```ts
function getByTestId(
  container: HTMLElement, //...
  text: TextMatch,
  options?: {
    exact?: boolean = true;
    normalizer?: NormalizerFn;
  }
): HTMLElement;
```

# custom queries

[См в документации](https://testing-library.com/docs/dom-testing-library/api-custom-queries)

# within

для поиска внутри контейнера

```js
import { within } from "@testing-library/dom";

const messages = document.getElementById("messages");
const helloMessage = within(messages).getByText("hello");
```
