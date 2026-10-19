# use

основная задача хука обрабатывать callback и в паре со Suspense регулировать отображение

```tsx
import { use } from "react";

interface Comment {
  id: string;
}

interface CommentsProps {
  commentsPromise: Promise<Comment[]>;
}

function Comments({ commentsPromise }: CommentsProps): JSX.Element {
  // `use` будет приостановлено до тех пор, пока обещание не разрешится.
  const comments = use(commentsPromise);
  return <>{comments.map((comment) => <p key={comment.id}>{String(comment)}</p>)}</>;
}

interface PageProps {
  commentsPromise: Promise<Comment[]>;
}

function Page({ commentsPromise }: PageProps): JSX.Element {
  // Когда `use` приостанавливается в Comments,
  // будет показана этот Suspense.
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Comments commentsPromise={commentsPromise} />
    </Suspense>
  );
}
```

Может также читать контекст. Можно вызывать внутри блоков условий

```tsx
interface ComponentProps {
  condition: boolean;
  promise: Promise<string>;
}

function Component({ condition, promise }: ComponentProps): JSX.Element {
  if (condition) {
    const data = use(promise);
    return <div>{data}</div>;
  }
  return <div>Ничего нет</div>;
}
```
