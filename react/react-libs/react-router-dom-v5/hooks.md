# useHistory

Хук для работы с history api

```jsx
import { useHistory } from "react-router-dom";

function HomeButton() {
  let history = useHistory();

  function handleClick() {
    history.push("/home");
  }

  return (
    <button type="button" onClick={handleClick}>
      Go home
    </button>
  );
}
```

history - объект с полями:

- length - (number) количество записей в истории
- action - (string) текущее действие
- location - (object) Объект локации с полями:
- - pathname - (string) URL
- - search - (string) строка запроса
- - hash - (string) URL hash
- - state - (object) - состояние переданное в предыдущий вызов
- push(path, [state]) - (function) добавит новую запись
- replace(path, [state]) - (function) заменит текущую запись
- go(n) - (function) назад на n шагов
- goBack() - (function) к предыдущей
- goForward() - (function) вперед по истории на один шаг
- block(prompt) - (function) выключает навигацию

# useLocation()

хук который предоставляет информацию о том, где мы находимся. Параметр state передается из хука useNavigate

![use-location](/assets/react/rrd-use-location.png)

# useMatch()

Возвращает данные соответствия о маршруте по заданному пути относительно текущего местоположения.

matchPath
сопоставляет шаблон пути маршрута с именем пути URL и возвращает информацию о совпадении. Это полезно, когда вам нужно вручную запустить алгоритм сопоставления маршрутизатора, чтобы определить, совпадает ли путь маршрута или нет. Он возвращает ноль, если шаблон не соответствует заданному имени пути. Хук useMatch использует эту функцию внутри, чтобы сопоставить путь маршрута относительно текущего местоположения.

# useNavigate()

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

хук useNavigation позволяет отследить состояние загрузки. Возможно следующе состояния: "idle" | "submitting" | "loading"

```tsx
import { useNavigation } from "react-router-dom";
export default function Root() {
  const navigation = useNavigation();
  return (
    <div className={navigation.state === "loading" ? "loading" : ""}> </div>
  );
}
```

# useParams

Возвращает объект в котором будут отображены ссылки

```jsx
// SinglePage.jsx;
<Route path={`/posts/:id/:params`} element={<SinglePage />} />;

const SinglePage = () => {
  const { id } = useParams();
  return <div>{id}</div>;
};
```

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

# useRouteMatch

позволяет провести соответствие между адресной строкой и и строкой переданной первым параметром

```js
//true: path:/1
const isMatch = useRouteMatch("path/:id"); // true
```

# useSearchParams()

```tsx
import { Link, useSearchParams } from "react-router-dom";
const Component = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const postQuery = searchParams.get("post"); //.../posts?post=abc&data=123 метод позволяет распарсить строку на значения
};
```
