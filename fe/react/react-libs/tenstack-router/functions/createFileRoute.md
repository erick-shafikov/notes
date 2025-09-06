# Route

```tsx
export const Route = createFileRoute("/posts/$postId")({
  component: PostComponent,
});

function PostComponent() {
  // получить параметры строки
  const { postId } = Route.useParams();
  return <div>Post {postId}</div>;
}
```
