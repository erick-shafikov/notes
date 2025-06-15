```tsx
import Form from "next/form";

export default function Page() {
  return (
    <>
      {/* если action - строка */}
      <Form action="/search" replace={false} scroll={true} prefetch={true}>
        <input name="query" />
        <button type="submit">Submit</button>
      </Form>
      {/* если action - функция */}
      <Form action="/search" replace={false} scroll={true} prefetch={true}>
        <input name="query" />
        <button type="submit">Submit</button>
      </Form>
    </>
  );
}
```
