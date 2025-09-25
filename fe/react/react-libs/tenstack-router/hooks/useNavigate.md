# useNavigate

Параметры:

- Принимает (объект):
- - from - строка
- Возвращает:
- - Функция с параметрами:
- - - Аргументы:
- - - - [NavigateOptions](../types/NavigateOptions.md)
- - - Возвращает:
- - - - Promise, разрешится при окончании навигации

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
