# методы vi.fn().method()

## getMockImplementation, mockImplementation, mockImplementationOnce, withImplementation

Вернет функцию которую замокировали

```ts
const mockFn = vi.fn().mockImplementation((apples: number) => apples + 1);
// or: vi.fn(apples => apples + 1);

const NelliesBucket = mockFn(0);
const BobsBucket = mockFn(1);

NelliesBucket === 1; // true
BobsBucket === 2; // true

mockFn.mock.calls[0][0] === 0; // true
mockFn.mock.calls[1][0] === 1; // true
```

```ts
const myMockFn = vi
  .fn()
  .mockImplementationOnce(() => true) // 1st call
  .mockImplementationOnce(() => false); // 2nd call

myMockFn(); // 1st call: true
myMockFn(); // 2nd call: false
```

## getMockName, mockName

вернет имя

## mockClear, mockReset, mockRestore

для сброса значений spyOn

```ts
const person = {
  greet: (name: string) => `Hello ${name}`,
};
const spy = vi.spyOn(person, "greet").mockImplementation(() => "mocked");
expect(person.greet("Alice")).toBe("mocked");
expect(spy.mock.calls).toEqual([["Alice"]]);

// очищаем историю вызовов
spy.mockClear();
expect(spy.mock.calls).toEqual([]);
expect(person.greet("Bob")).toBe("mocked");
expect(spy.mock.calls).toEqual([["Bob"]]);
```

```ts
const person = {
  greet: (name: string) => `Hello ${name}`,
};
const spy = vi.spyOn(person, "greet").mockImplementation(() => "mocked");
expect(person.greet("Alice")).toBe("mocked");
expect(spy.mock.calls).toEqual([["Alice"]]);

// clear call history and restore spied object method
spy.mockRestore();
expect(spy.mock.calls).toEqual([]);
expect(person.greet).not.toBe(spy);
expect(person.greet("Bob")).toBe("Hello Bob");
expect(spy.mock.calls).toEqual([]);
```

## mockRejectedValue, mockRejectedValueOnce

```ts
const asyncMock = vi.fn().mockRejectedValue(new Error("Async error"));

await asyncMock(); // throws Error<'Async error'>
```

## mockResolvedValue, mockResolvedValueOnce

для асинхронного возврата значения

```ts
const asyncMock = vi.fn().mockResolvedValue(42);

await asyncMock(); // 42
```

```ts
// чейнинг значений
const asyncMock = vi
  .fn()
  .mockResolvedValue("default")
  .mockResolvedValueOnce("first call")
  .mockResolvedValueOnce("second call");

await asyncMock(); // first call
await asyncMock(); // second call
await asyncMock(); // default
await asyncMock(); // default
```

## mockReturnThis

для возврата this

```ts
spy.mockImplementation(function () {
  return this;
});
```

## mockReturnValue, mockReturnValueOnce

для возврата значения

```ts
const mock = vi.fn();
mock.mockReturnValue(42);
mock(); // 42
mock.mockReturnValue(43);
mock(); // 43

// mockReturnValueOnce для чейнинга
const myMockFn = vi
  .fn()
  .mockReturnValue("default")
  .mockReturnValueOnce("first call")
  .mockReturnValueOnce("second call");

// 'first call', 'second call', 'default', 'default'
console.log(myMockFn(), myMockFn(), myMockFn(), myMockFn());
```

# поля mock.value

## mock.calls, mock.lastCall

- calls - вернет массив аргументов вызова
- lastCall - вернет массив аргумент последнего вызова

```ts
const fn = vi.fn();

fn("arg1", "arg2");
fn("arg3");

fn.mock.calls ===
  [
    ["arg1", "arg2"], // first call
    ["arg3"], // second call
  ];
```

## mock.results, mock.settledResults

результаты

## mock.invocationCallOrder

прядок вызова

```ts
const fn1 = vi.fn();
const fn2 = vi.fn();

fn1();
fn2();
fn1();

fn1.mock.invocationCallOrder === [1, 3];
fn2.mock.invocationCallOrder === [2];
```

## mock.contexts

```ts
const fn = vi.fn();
const context = {};

fn.apply(context);
fn.call(context);

fn.mock.contexts[0] === context;
fn.mock.contexts[1] === context;
```

## mock.instances
