# branded-types

Проблема - postId и authorId string

```ts
async function getCommentsForPost(postId: string, authorId: string) {...}

// перепутали id пользователя и поста
getCommentsForPost(user.id, post.id) {...} // TS это пропустит
```

```ts
type Kilometer = number & {
  __brand: "Kilometer";
};
```

```ts
declare const __brand: unique symbol;

type Branded<T, UniqueKey> = T & { [__brand]: UniqueKey };
```

```ts
declare const kmBrand: unique symbol;

type Branded<T, UniqueKey> = T & { [__brand]: never };
```

- Мы принимаем тип T и расширяем его уникальным свойством \_\_brand.
- unique symbol равен только самому себе, мы можем отличать “особые” примитивы от обычных.
- Мы помещаем brand в вычисляемое свойство, чтобы сделать его невидимым. Это означает, что доступ к authorId.brand невозможен.
- Мы присваиваем свойству \_\_brand уникальный ключ, чтобы различать разные брендированные типы. Например, UserID отличается от PostID.

использование

```ts

type UserID = Branded<string, "UserID">

type PostID = Branded<string, "PostID">

const userId: UserID = "123" as UserID;

const postId: PostID = "456" as PostID;

async function getCommentsForPost(postId: PostID, authorId: UserID) {...}

getCommentsForPost(userId, postId); // Ошибка: Аргументы типа 'UserID' и 'PostID' в данном контексте несовместимы.
```

# Фабрика

```ts
type Brand<T, B extends symbol> = T & { [key in B]: never };

type BrandNumber<B extends symbol> = Brand<number, T>;
```
