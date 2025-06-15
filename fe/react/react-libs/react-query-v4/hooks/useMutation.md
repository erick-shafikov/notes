```tsx
const {
  //The last successfully resolved data for the query.
  data, // (variables: TVariables, { onSuccess, onSettled, onError }) => void, variables: TVariables - переданные в mutationFn
  error, // null | TError
  isError, // boolean
  isIdle, // boolean
  isLoading, // boolean
  isPaused, // boolean
  isSuccess, // boolean
  failureCount, // number
  failureReason, // null | TError
  mutate,
  mutateAsync, //(variables: TVariables, { onSuccess, onSettled, onError }) => Promise<TData> вернет промис, который можно await
  reset, //() => void
  status, //idle | loading | error | success
} = useMutation({
  mutationFn, //mutationFn: (variables: TVariables) => Promise<TData> -функция, которая возвращает промис,
  // variables - параметры переданные в mutate
  cacheTime, //number | Infinity
  mutationKey, //unknown[]
  networkMode, //'online' | 'always' | 'offlineFirst'
  onError, // (err: TError, variables: TVariables, context?: TContext) => Promise<unknown> | unknown
  onMutate, //(variables: TVariables) => Promise<TContext | void> | TContext | void
  onSettled, //(data: TData, error: TError, variables: TVariables, context?: TContext) => Promise<unknown> | unknown
  onSuccess, // (data: TData, variables: TVariables, context?: TContext) => Promise<unknown> | unknown
  retry, //boolean | number | (failureCount: number, error: TError) => boolean
  retryDelay, //number | (retryAttempt: number, error: TError) => number
  useErrorBoundary, //undefined | boolean | (error: TError) => boolean
  meta, // Record<string, unknown>
});

mutate(variables, {
  onError, //(err: TError, variables: TVariables, context: TContext | undefined) => void
  onSettled, //(data: TData | undefined, error: TError | null, variables: TVariables, context: TContext | undefined) => void
  onSuccess, // (data: TData, variables: TVariables, context: TContext) => void
});
```
