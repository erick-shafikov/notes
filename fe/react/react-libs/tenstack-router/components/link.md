# Link компонент

пример ссылки с параметрами

```tsx
const link = () => (
  <Link
    to="/blog/post/$postId"
    params={{
      postId: "my-first-blog-post",
    }}
    //удаление параметров
    params={{ category: undefined }}
    params={{}}
    //необязательные
    params={{ category: undefined, slug: undefined }}
    //функционально обновление, императивная навигация
    params={(prev) => ({ ...prev, category: undefined })}
    //
    //параметры поиска
    search={{
      query: "tanstack",
    }}
    // обновить точечно один параметр
    search={(prev) => ({
      ...prev,
      page: prev.page + 1,
    })}
    //расширенный поиск
    ///shop?pageIndex=3&includeCategories=%5B%22electronics%22%2C%22gifts%22%5D&sortBy=price&desc=true
    search={{
      pageIndex: 3,
      includeCategories: ["electronics", "gifts"],
      sortBy: "price",
      desc: true,
    }}
    //к определенному id
    hash="section-1"
  >
    Blog Post
  </Link>
);
```

работа с префиксами

```tsx
const LinkWithPrefix = () => (
  <Link to="/files/prefix{-$name}.txt" params={{ name: undefined }}>
    Default File
  </Link>
);
```

```tsx
const StyledLink = () => (
  <Link
    to="/blog/post/$postId"
    params={{
      postId: "my-first-blog-post",
    }}
    activeProps={{
      style: {
        fontWeight: "bold",
      },
    }}
    activeOptions={{
      exact: true,
      includeHash: false,
      includeSearch: false,
      explicitUndefined: true,
    }}
  >
    Section 1
  </Link>
);
```

специальные параметры to:

- . - перезагрузка текущего
- .. - назад на один

# вспомогательные функции

Функции, для работы с компонентом Link:

- [linkOptions - позволяет создать пропсы для Link](../functions/linkOptions.md)
- [createLink - позволяет создать компонент Link](../functions/createLink.md)
- [дженерик-функция, которая поможет вывести тип пропсов для пользовательского компонента Link](../types/ValidateLinkOptions.md)

# маскировка путей

```tsx
// скроет modal
const MaskLinkComponent = () => (
  <Link
    to="/photos/$photoId/modal"
    params={{ photoId: 5 }}
    mask={{
      to: "/photos/$photoId",
      params: {
        photoId: 5,
      },
    }}
  >
    Open Photo
  </Link>
);
// навигация
function onOpenPhoto() {
  navigate({
    to: "/photos/$photoId/modal",
    params: { photoId: 5 },
    mask: {
      to: "/photos/$photoId",
      params: {
        photoId: 5,
      },
    },
  });
}
```
