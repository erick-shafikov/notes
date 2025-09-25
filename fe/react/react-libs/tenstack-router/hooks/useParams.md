# useParams

Параметры:

- Принимает (объект):
- - strict - true, если false from будет проигнорирован
- - shouldThrow - если false не прокинет ошибку
- - select - (params: AllParams) => TSelected
- - structuralSharing - bool
- Возвращает:
- - объект с параметрами

```tsx
function PostComponent() {
  // получить параметры
  const { postId } = useParams({
    strict: false, //получить параметры из неопределенного местоположение
  });

  return <div>Post {postId}</div>;
}
```
