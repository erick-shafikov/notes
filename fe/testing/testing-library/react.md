# render

```ts
function render(
  ui: React.ReactElement<any>,
  options?: {
    container: Element;
    baseElement: Element = document.body;
    hydrate: boolean;
    onCaughtError: () => void;
    onRecoverableError: () => void;
    wrapper: React.Component;
    queries: CustomQueries;
  }
): RenderResult;
// возвращает
const {
  // все функции запроса
  getByLabelText,
  queryAllByTestId,
  // и др
  container,
  baseElement,
  debug,
  rerender,
  // удалит элемент
  unmount,
  asFragment,
} = render(<Component />);
```

rerender

```js
import { render } from "@testing-library/react";

const { rerender } = render(<NumberDisplay number={1} />);

// re-render the same component with different props
rerender(<NumberDisplay number={2} />);
```

Применение:

```js
import { render } from "@testing-library/react";
import "@testing-library/jest-dom";

test("renders a message", () => {
  const { asFragment, getByText } = render(<Greeting />);
  expect(getByText("Hello, world!")).toBeInTheDocument();
  expect(asFragment()).toMatchInlineSnapshot(`
    <h1>Hello, World!</h1>
  `);
});
```

# cleanup

Удалит дерево react компонентов после монтирования

```js
import { cleanup, render } from "@testing-library/react";
import test from "ava";

test.afterEach(cleanup);

test("renders into document", () => {
  render(<div />);
  // ...
});

// ... more tests ...
```

# renderHook

```js
import { renderHook } from "@testing-library/react";

test("returns logged in user", () => {
  const { result } = renderHook(() => useLoggedInUser());
  expect(result.current).toEqual({ name: "Alice" });
});
```

# configure

```js
import { configure } from "@testing-library/react";

configure({ reactStrictMode: true });
```
