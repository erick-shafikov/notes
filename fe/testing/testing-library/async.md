# waitFor

```ts
function waitFor<T>(
  callback: () => T | Promise<T>,
  options?: {
    container?: HTMLElement;
    timeout?: number;
    interval?: number;
    onTimeout?: (error: Error) => Error;
    mutationObserverOptions?: MutationObserverInit;
  }
): Promise<T>;
```

Пример

```js
// будет ждать до ошибки
await waitFor(() => expect(mockAPI).toHaveBeenCalledTimes(1));
```

# waitForElementToBeRemoved

```ts
function waitForElementToBeRemoved<T>(
  callback: (() => T) | T,
  options?: {
    container?: HTMLElement;
    timeout?: number;
    interval?: number;
    onTimeout?: (error: Error) => Error;
    mutationObserverOptions?: MutationObserverInit;
  }
): Promise<void>;
```

# появление и исчезновение элементов

через асинхронные операции поиска findBy

```js
test("movie title appears", async () => {
  // element is initially not present...
  // wait for appearance and return the element
  const movie = await findByText("the lion king");
});
```

или через waitFor

```js
test("movie title appears", async () => {
  await waitFor(() => {
    expect(getByText("the lion king")).toBeInTheDocument();
  });
});
```

исчезновение

```js
test("movie title no longer present in DOM", async () => {
  // element is removed
  await waitForElementToBeRemoved(() => queryByText("the mummy"));
});
```
