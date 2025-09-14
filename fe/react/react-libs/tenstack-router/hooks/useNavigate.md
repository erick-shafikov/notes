# useNavigate

```tsx
function Component() {
  const navigate = useNavigate({ from: "/posts/$postId" });

  const handleSubmit = async (e: FrameworkFormEvent) => {
    e.preventDefault();

    if (response.ok) {
      navigate({ to: "/posts/$postId", params: { postId } });
    }
  };
}
```
