# useFormStatus

позволяет узнать состояние формы без использование контекста

```js
import { useFormStatus } from "react-dom";

function DesignButton() {
  const { pending } = useFormStatus();
  return <button type="submit" disabled={pending} />;
}
```
