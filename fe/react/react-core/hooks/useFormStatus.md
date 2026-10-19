# useFormStatus

позволяет узнать состояние формы без использование контекста

```tsx
import { useFormStatus } from "react-dom";
import { useState } from "react";

function SubmitButton(): JSX.Element {
  const { pending, data } = useFormStatus();

  // Можно использовать data для дополнительных проверок
  const comment = data?.get("comment") || "";

  return (
    <button type="submit" disabled={pending}>
      {pending ? "Отправка..." : "Отправить"}
    </button>
  );
}

function CommentForm(): JSX.Element {
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(formData: FormData): Promise<void> {
    try {
      const response = await fetch("/api/comments", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Ошибка сервера");
      setError(null); // Сбрасываем ошибку при успехе
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <form action={handleSubmit}>
      <input name="comment" placeholder="Ваш комментарий" />
      <SubmitButton />
      {error && <p style={{ color: "red" }}>{error}</p>}
    </form>
  );
}
```
