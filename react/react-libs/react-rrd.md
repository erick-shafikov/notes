# Настройка

Установка
yarn add react-router-dom
npm install react-router-dom@6

```jsx
// index.js;
const appLication = (
  //оборачиваем приложение в роутинг
  <BrowserRouter>
    <App />
  </BrowserRouter>
);
ReactDOM.render(appLication, document.getElementById("root"));
```

# Компоненты

## `<Link>` и `<NavLink>`

`<Link>` - это элемент, который позволяет пользователю переходить на другую страницу, щелкая или нажимая на нее. В response-router-dom `<Link>` отображает доступный элемент `<a>` с реальным href, указывающим на ресурс, на который он ссылается. Это означает, что такие вещи, как щелчок правой кнопкой мыши по `<Link>`, работают должным образом. Вы можете использовать `<Link reloadDocument>`, чтобы пропустить маршрутизацию на стороне клиента и позволить браузеру обрабатывать переход в обычном режиме (как если бы это был `<a href>`).

`<NavLink>` - автоматически при переходе на активную ссылку добавляет к активной вкладке класс active и isPending при загрузке

пропсы:

- **to** направление
- className = ({isActive, isPending}) => style функция, которая позволяет стилизовать активную/неактивную ссылку
- style = ({isActive, isPending, isTransitioning}) => {style obj} возвращает объект стилей
- end: Boolean - позволяет провести соответствие между вложенными роутами и в соответствии с этим определить isActive

```js
<NavLink to="/tasks" />	// path: '/tasks'	=> true
<NavLink to="/tasks" />	// path: '/tasks/123'	=> true
<NavLink to="/tasks" end />	// path: '/tasks'	=> true
<NavLink to="/tasks" end />	// path: '/tasks/123'	=> false
<NavLink to="/tasks/" end />	// path: '/'tasks'	=> false
<NavLink to="/tasks/" end />	// path: '/tasks/''	=> true
```

- createSensitive - различать или нет регистр зависимые url
- reloadDocument - поведение обычного a на клиентской стороне
- unstable_viewtransition - для работы с View Transition

```tsx
<NavLink to="/" className={({isActive, isPending})=> isActive? 'active-link' : '' }> Home</NavLink>
//для подсвечивания каждой ссылки индивидуально, можно вынести логику в отдельную функцию

<NavLink to="/" _style={({isActive}) => ({backgroundColor: isActive? 'cyan' : 'white'})}>Home</NavLink>
//c помощью проп style
```

### Кастомный <NavLink>

```js
import { Outlet } from "react-router-dom";
import CustomLink from "./CustomLink";
import "./../styles/styles.css";
const Layout = () => {
  return (
    <>
      <header>
        <CustomLink to="/">Home</CustomLink>
        <CustomLink to="/posts">Blog</CustomLink>
        <CustomLink to="/about">About</CustomLink>
      </header>
      <Outlet />
      <footer>2022</footer>
    </>
  );
};
export default Layout;
```

```js
import { Link, useMatch } from "react-router-dom"; //useMatch хук возвращает true если совпадает путь
const CustomLink = ({ children, to, ...props }) => {
  //для того что бы передавать дочерние элементы
  const match = useMatch(to);
  console.log({ match });
  return (
    <Link
      to={to}
      _style={{
        color: match ? "red" : "white",
      }}
      {...props}
    >
      {children}
    </Link>
  );
};
export default CustomLink;
```

## Navigate

Компонент позволяет произвести редирект

```js
declare function Navigate(props: NavigateProps): null;

interface NavigateProps {
  to: To;
  replace?: boolean;
  state?: any;
  relative?: RelativeRoutingType;
}

import * as React from "react";
import { Navigate } from "react-router-dom";

const LoginForm = () => {
  return <Navigate to="/dashboard" replace={true} />;
};
```

## Outlet

```tsx
import { NavLink, Outlet } from "react-router-dom";
import styles from "./../styles/styles.module.css";
const Layout = () => {
  return (
    <header>
    <NavLink to="/">Home</NavLink>       
    <NavLink to="/posts">Blog</NavLink>       
    <NavLink to="/about">About</NavLink>
    </header>
    <Outlet />
      //В данный компонент будут рендерится пропы elements
      <footer>2022</footer>
  );
};
export default Layout;
```

## Routes

```jsx
import {
  createRoutesFromElements,
  createBrowserRouter,
} from "react-router-dom";
const router = createBrowserRouter(
  createRoutesFromElements(
    <Route
      path="/"
      element={<Root />}
      loader={rootLoader}
      action={rootAction}
      errorElement={<ErrorPage />}
    >
      <Route errorElement={<ErrorPage />}>
        <Route index element={<Index />} />
        <Route
          path="contacts/:contactId"
          element={<Contact />}
          loader={contactLoader}
          action={contactAction}
        />
        <Route
          path="contacts/:contactId/edit"
          element={<EditContact />}
          loader={contactLoader}
          action={editAction}
        />
        <Route path="contacts/:contactId/destroy" action={destroyAction} />
      </Route>
    </Route>
  )
);
```

```tsx
<Route> {/*как в выражении if; если его путь совпадает с текущим URL-адресом, он отображает свой элемент*/}

<Routes>
  <Route path="/" element={<h1>Home Page</h1>} /> {/*Вариант с рендером элемента внутри Route*/}
  <Route path='/cars' element={<Cars />}/> {/*Вариант с рендером компонента*/}
</Routes>

<Routes>  {/*компонент для оборачивания роутов*/}
<Route >  {/*компонент, который направляет конкретный роут на другую страницу*/}

<Link to="/">Home</Link>{/*компонент Link замена <a> - который не вызывает перезагрузки страницы при каждом переходе на новую*/}

<Switch>  {/* убран */}
{/* exact - поведение по дефолту */}
```

# Функции

## action

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

## createBrowserRouter

Организация роутинга с помощью объекта

```tsx
//подключение
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
        <RouterProvider router={router} fallbackElement={<LoadingElement />} /> {" "}
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

### basename

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

## loader

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

# Хуки

## useLocation()

хук который предоставляет информацию о том, где мы находимся. Параметр state передается из хука useNavigate

![use-location](/assets/react/rrd-use-location.png)

## useMatch()

useMatch
Возвращает данные соответствия о маршруте по заданному пути относительно текущего местоположения.

matchPath
сопоставляет шаблон пути маршрута с именем пути URL и возвращает информацию о совпадении. Это полезно, когда вам нужно вручную запустить алгоритм сопоставления маршрутизатора, чтобы определить, совпадает ли путь маршрута или нет. Он возвращает ноль, если шаблон не соответствует заданному имени пути. Хук useMatch использует эту функцию внутри, чтобы сопоставить путь маршрута относительно текущего местоположения.

## useNavigate()

Хук useNavigate возвращает функцию, которая позволяет вам перемещаться программно, например, после отправки формы. Замена useHistory

Либо передайте значение To (того же типа, что и `<Link to>`) с необязательным вторым аргументом {replace, state}, либо Передайте желаемую дельту в стеке истории. Например, навигация (-1) эквивалентна нажатию кнопки возврата.

```tsx
import { useNavigate } from "react-router-dom";
function SignupForm() {
  let navigate = useNavigate();

  async function handleSubmit(event) {
    event.preventDefault();
    await submitForm(event.target);
    navigate("../success", { replace: true });
  }

  return <form onSubmit={handleSubmit}>{/* ... */}</form>;
}
```

Принимает 2 параметра: первый ссылка или число шагов назад (-1 – вернуться на одну страницу назад)
Также может принимать две опции replace и state
replace : true – движение без записи в историю
replace : false – движение c записи в историю

```tsx
const SinglePage = () => {
  const navigate = useNavigate();
  const goBack = () => navigate(-1); //перекинет на 1 страницу назад
  const goHome = () => navigate("/", { replace: true }); //перекинет на главную, с записью в историю
  const goBack = () => navigate("/posts", { state: 123 }); //state отловит useLocation()

  return (
    <div>
      <button onClick={goBack}>Go back</button>
      <button onClick={goHome}>Go Home</button>
    </div>
  );
};
```

хук useNavigation позволяет отследить состояние загрузки. Возможно следующе состояния:
"idle" | "submitting" | "loading"

```tsx
import { useNavigation } from "react-router-dom";
export default function Root() {
  const navigation = useNavigation();
  return (
    <div className={navigation.state === "loading" ? "loading" : ""}> </div>
  );
}
```

## useParams

Возвращает объект в котором будут отображены ссылки

```jsx
// SinglePage.jsx;
<Route path={`/posts/:id/:params`} element={<SinglePage />} />;

const SinglePage = () => {
  const { id } = useParams();
  return <div>{id}</div>;
};
```

![params](/assets/react/rrd-params.png)

Хук useParams возвращает объект пар ключ / значение динамических параметров из текущего URL-адреса, которому соответствует `<Route path>`. Дочерние маршруты наследуют все параметры от своих родительских маршрутов.

```tsx
import * as React from "react";
import { Routes, Route, useParams } from "react-router-dom";
function ProfilePage() {
  // Get the userId param from the URL.
  let { userId } = useParams(); // ...
}
function App() {
  return (
    <Routes>
      {" "}
      <Route path="users">
        <Route path=":userId" element={<ProfilePage />} />
        <Route path="me" element={} />
      </Route>
    </Routes>
  );
}
```

## useRouteMatch

позволяет провести соответствие между адресной строкой и и строкой переданной первым параметром

```js
//true: path:/1
const isMatch = useRouteMatch("path/:id"); // true
```

## useSearchParams()

```tsx
import { Link, useSearchParams } from "react-router-dom";
const Component = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const postQuery = searchParams.get("post"); //.../posts?poast=abc&data=123 метод позволяет распарсить строку на значения
};
```
