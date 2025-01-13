# `<Link>` и `<NavLink>`

`<Link>` - это элемент, который позволяет пользователю переходить на другую страницу, щелкая или нажимая на нее. В response-router-dom `<Link>` отображает доступный элемент `<a>` с реальным href, указывающим на ресурс, на который он ссылается. Это означает, что такие вещи, как щелчок правой кнопкой мыши по `<Link>`, работают должным образом. Вы можете использовать `<Link reloadDocument>`, чтобы пропустить маршрутизацию на стороне клиента и позволить браузеру обрабатывать переход в обычном режиме (как если бы это был `<a href>`).

`<NavLink>` - автоматически при переходе на активную ссылку добавляет к активной вкладке класс active и isPending при загрузке

пропсы:

- to направление
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

## Кастомный <NavLink>

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

# Navigate

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

# Outlet

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

# Routes

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
const Comp = () => {
  return (
    //<Routes> -компонент для оборачивания роутов
    //<Route> -как в выражении if; если его путь совпадает с текущим URL-адресом, он отображает свой элемент
    //<Route > -компонент, который направляет конкретный роут на другую страницу
    //<Link to="/">Home</Link> - компонент Link замена <a> - который не вызывает перезагрузки страницы при каждом переходе на новую
    //<Switch> -  убран exact - поведение по дефолту

    <Routes>
      <Route path="/" element={<h1>Home Page</h1>} />{" "}
      {/*Вариант с рендером элемента внутри Route*/}
      <Route path="/cars" element={<Cars />} /> {/*Вариант с рендером компонента*/}
    </Routes>
  );
};
```
