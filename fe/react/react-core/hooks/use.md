# use

основная задача хука обрабатывать callback и в паре со Suspense регулировать отображение

```js
import { use } from "react";

function Comments({ commentsPromise }) {
  // `use` будет приостановлено до тех пор, пока обещание не разрешится.
  const comments = use(commentsPromise);
  return comments.map((comment) => <p key={comment.id}>{comment}</p>);
}

function Page({ commentsPromise }) {
  // Когда `use` приостанавливается в Comments,
  // будет показана этот Suspense.
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Comments commentsPromise={commentsPromise} />
    </Suspense>
  );
}
```

Может также читать контекст
