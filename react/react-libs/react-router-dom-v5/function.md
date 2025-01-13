# action

Для обновления данный, например при редактирования

```js
import { Form, useLoaderData, redirect } from "react-router-dom";
import { updateContact } from "../contacts";

//метод для обновления
export async function action({ request, params }) {
  const formData = await request.formData();
  const updates = Object.fromEntries(formData);
  await updateContact(params.contactId, updates);
  return redirect(`/contacts/${params.contactId}`);
}
//метод для поиска
export async function loader({ request }) {
  const url = new URL(request.url);
  const q = url.searchParams.get("q");
  const contacts = await getContacts(q);
  return { contacts };
}

//добавляем в роут
// {
//   path: "contacts/:contactId/edit",
//   element: <EditContact />,
//   loader: contactLoader,
//   action: editAction,
// },
```

# createBrowserRouter

Организация роутинга с помощью объекта

```tsx
//подключение
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} fallbackElement={<LoadingElement />} />
  </React.StrictMode>
);

//создания роутинга
const router = createBrowserRouter([
  {
    path: "/", //путь
    element: <Root />, //элемент для рендера
    errorElement: <ErrorPage />, //при ошибке при переходе
    loader: rootLoader, //загрузка данных см ↳ loader
    children: [
      //вложенные роуты для Outlet
      {
        path: "contacts/:contactId",
        element: <Contact />,
      },
      { index: true, element: <Index /> }, //при ссылке на пустой Outlet можно добавить элемент по умолчанию <Index />
    ],
  },
]);
```

## basename

Позволяет добавить к url базовую часть

```tsx
createBrowserRouter(routes, {
  basename: "/app",
});
<Link to="/" />; // results in <a href="/app" />
createBrowserRouter(routes, {
  basename: "/app/",
});
<Link to="/" />; // results in <a href="/app/" />
```

# loader

```tsx
// функция для взаимодействия с API
export async function loader() {
  const contacts = await getContacts(); //метод для асинхронного запроса
  return { contacts };
}

// в компоненте
export default function Root() {
  const { contacts } = useLoaderData(); //достаем данные из хука
}

// Для работы с параметрами строки

// {
//   path: "contacts/:contactId",
//   element: <Contact />,
//   loader: contactLoader, //логика функции loader описана ниже
// }

export async function loader({ params }) {
  const contact = await getContact(params.contactId); //функция для взаимодействия с api (в зависимости от id загружаем нужные данные)
  return { contact };
}
```
