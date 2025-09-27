# Link компонент

пример ссылки с параметрами:

- [LinkProps](../types/LinkProps.md)

```tsx
const link = () => (
  <Link
    //PARAMS
    //передача динамических параметров
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
    //SEARCH PARAMS
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

# активная ссылка

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

рендер-пропс подход

```tsx
const link = (
  <Link to="/blog/post">
    {({ isActive }) => {
      return (
        <>
          <span>My Blog Post</span>
          <icon className={isActive ? "active" : "inactive"} />
        </>
      );
    }}
  </Link>
);
```

# специальные параметры to:

- . - перезагрузка текущего
- .. - назад на один
- компонент Route

```tsx
import { Route as aboutRoute } from "./routes/about.tsx";

function Comp() {
  return <Link to={aboutRoute.to}>About</Link>;
}

//или

const link = (
  <Link from={aboutRoute.fullPath} to="../categories">
    Categories
  </Link>
);
```

# вспомогательные функции

Функции, для работы с компонентом Link:

- [linkOptions - позволяет создать пропсы для Link](../functions/linkOptions.md)
- [createLink - позволяет создать компонент Link](../functions/createLink.md)
- [дженерик-функция, которая поможет вывести тип пропсов для пользовательского компонента Link](../types/ValidateLinkOptions.md)

# маскировка путей

- [creteRouteMask](../functions/createRouteMask.md)

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
