# useMutation

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
