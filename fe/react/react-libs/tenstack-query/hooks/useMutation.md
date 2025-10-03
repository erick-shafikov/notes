# useMutation

Параметры:

# принимает

два параметра, второй - контекст, первый объект с полями:

## mutationFn

```ts
type MutationFn = (
  variables: TVariables,
  context: MutationFunctionContext
) => Promise<TData>;
```

[MutationFunctionContext можно использовать](#использование-meta-полей)

## gcTime

number | Infinity - время в кеше (максимум 24 дня)

## meta

Произвольные данные прокидываемые в запрос

```ts
type Meta = Record<string, unknown>;
```

## mutationKey

```ts
type MutationKey = unknown[];
```

## networkMode

'online' | 'always' | 'offlineFirst'

## onError

```ts
type onError = (
  err: TError,
  variables: TVariables,
  onMutateResult: TOnMutateResult | undefined,
  context: MutationFunctionContext
) => Promise<unknown> | unknown;
```

## onMutate

```ts
type onMutate = (
  variables: TVariables
) => Promise<TOnMutateResult | void> | TOnMutateResult | void;
```

## onSettled

```ts
type onSettled = (
  data: TData,
  error: TError,
  variables: TVariables,
  onMutateResult: TOnMutateResult | undefined,
  context: MutationFunctionContext
) => Promise<unknown> | unknown;
```

## onSuccess

```ts
type OnSuccess = (
  data: TData,
  variables: TVariables,
  onMutateResult: TOnMutateResult | undefined,
  context: MutationFunctionContext
) => Promise<unknown> | unknown;
```

## retry

```ts
type Retry = boolean | number | (failureCount: number, error: TError) => boolean
```

## retryDelay

```ts
type RetryDelay = number | (retryAttempt: number, error: TError) => number
```

## scope

мутации с одним id будут идти последовательно

```ts
type Scope = { id: string };
```

## throwOnError

если true пробросит ошибку до ближайшего errorBoundary

```ts
type ThrowOnError = undefined | boolean | (error: TError) => boolean
```

<!--  -->

# возвращает

## data

undefined - результат мутации

## error

null | TError

## isError

boolean

## isIdle

boolean

## isPending

boolean

## isPaused

boolean

## isSuccess

boolean

## failureCount

number - 0 когда мутация успешная

## failureReason

null | TError - причина перезапуска мутации

## mutate

```ts
type Mutate = (
  variables: TVariables,
  options: {
    onSuccess: (
      data: TData,
      variables: TVariables,
      onMutateResult: TOnMutateResult | undefined,
      context: MutationFunctionContext
    ) => void;
    onSettled: (
      data: TData | undefined,
      error: TError | null,
      variables: TVariables,
      onMutateResult: TOnMutateResult | undefined,
      context: MutationFunctionContext
    ) => void;
    onError: (
      err: TError,
      variables: TVariables,
      onMutateResult: TOnMutateResult | undefined,
      context: MutationFunctionContext
    ) => void;
  }
) => void;
```

## mutateAsync

как и mutate но вернет промис

## reset

сброс мутации

```ts
type Reset = () => void;
```

## status

idle | pending | error | success

## submittedAt

0 - timestamp отправки мутации

## variables

# использование meta-полей

можно прописать автоматическую ре-валидацию queryClient при мутации

```js
export const useDeleteContract() => useMutation({
  mutationFn: (contractId) => client.deleteContract(contractId),
  meta: { invalidatesQueries: ['Contracts'] },
})
```

```js
const queryClient = new QueryClient({
  onSettled: (_data, _error, _variables, _context, mutation) => {
    if (mutation.meta?.invalidateQuery) {
      queryClient.invalidateQueries({
        queryKey: mutation.meta?.invalidatesQueries,
      });
    }
  },
});
```
