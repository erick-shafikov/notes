## useActionState

это хук, который позволяет обновлять состояние на основе результата действия формы.

```tsx
const [state, formAction, isPending] = useActionState(fn, initialState, permalink?);
//fn - функция при вызове отправки
//initialState - начальное состояние формы
// state - текущее состояние
// formAction - действие
// permalink - ссылка для изменения формы
```

Применение отображать результат отправки формы

```tsx
import { useActionState } from "react";

async function increment(previousState: number, formData: FormData): Promise<number> {
  return previousState + 1;
}

function StatefulForm(): JSX.Element {
  const [state, formAction] = useActionState(increment, 0);
  return (
    <form>
      {state}
      <button formAction={formAction}>Increment</button>
    </form>
  );
}
```

с асинхронным состоянием

```tsx
interface ChangeNameProps {
  name: string;
  setName: (name: string) => void;
}

function ChangeName({ name, setName }: ChangeNameProps): JSX.Element {
  const [error, submitAction, isPending] = useActionState(
    async (previousState: string | null, formData: FormData): Promise<string | null> => {
      const error = await updateName(formData.get("name") as string);
      if (error) {
        return error;
      }
      redirect("/path");
      return null;
    },
    null
  );

  return (
    <form action={submitAction}>
      <input type="text" name="name" />
      <button type="submit" disabled={isPending}>
        Update
      </button>
      {error && <p>{error}</p>}
    </form>
  );
}
```
