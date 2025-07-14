# useFormStatus

позволяет узнать состояние формы без использование контекста

```js
import { useFormStatus } from "react-dom";
import { useState } from "react";

function SubmitButton() {
  const { pending, data } = useFormStatus();

  // Можно использовать data для дополнительных проверок
  const comment = data?.get("comment") || "";

  return (
    <button type="submit" disabled={pending}>
      {pending ? "Отправка..." : "Отправить"}
    </button>
  );
}

function CommentForm() {
  const [error, setError] = useState(null);

  async function handleSubmit(formData) {
    try {
      const response = await fetch("/api/comments", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Ошибка сервера");
      setError(null); // Сбрасываем ошибку при успехе
    } catch (err) {
      setError(err.message);
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
