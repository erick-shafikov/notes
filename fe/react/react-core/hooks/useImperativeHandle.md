## useImperativeHandle

useImperativeHandle(ref, createHandle, dependencies?) – позволяет добавить объекту ref дополнительный функционал, где createHandle это функция, которая возвращает объект с полями, которые в последствии можно будет вызывать на объекте ref.current (смешанные скролл и фокусы)

Позволяет ограничить доступ к ссылке

```tsx
import { forwardRef, useRef, useImperativeHandle } from "react";

interface MyInputHandle {
  focus: () => void;
  scrollIntoView: () => void;
}

const MyInput = forwardRef<MyInputHandle, React.InputHTMLAttributes<HTMLInputElement>>(
  function MyInput(props, ref) {
  const inputRef = useRef<HTMLInputElement>(null);

  //у родительского компонента не будет ссылки непосредственно на ссылку элемента, а только на методы focus, scrollIntoView
  useImperativeHandle(
    ref,
    () => {
      return {
        focus() {
          inputRef.current?.focus();
        },
        scrollIntoView() {
          inputRef.current?.scrollIntoView();
        },
      };
    },
    []
  );

  return <input {...props} ref={inputRef} />;
});
```
