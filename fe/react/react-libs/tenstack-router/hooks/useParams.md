```tsx
function PostComponent() {
  // получить параметры
  const { postId } = useParams({
    strict: false, //получить параметры из неопределенного местоположение
  });

  return <div>Post {postId}</div>;
}
```
