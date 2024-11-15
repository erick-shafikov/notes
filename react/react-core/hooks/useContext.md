# useContext

```js
//компонент отвечающий за контекст
import { useContext, createContext } from "react";

const AlertContext = createContext(null); //создаем объект контекста

export const AlertProvider = ({ children }) => {
  //обертка AlertProvider для всего приложения
  const [alert, setAlert] = useState(false); //логика
  const toggle = () => setAlert(!alert);

  return (
    <AlertContext.Provider value={{ visible: alert, toggle }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlert = () => useContext(AlertContext); //функция для использования контекста в компонентах
```

```js
import Alert from "./Alert";
import { AlertProvider } from "./Alert/AlertContext";
import Main from "./Main";
//главное приложение ренедерит два компоненте
function UseContextHook() {
  return (
    <AlertProvider>
      <div className="container pt-3">
        <Alert />
        <Main />
      </div>
    </AlertProvider>
  );
}
export default UseReducerHook;
```

```js
//компоненте Main
import { useAlert } from "./Alert/AlertContext";

export default function Main() {
  const { toggle } = useAlert(); //достаем из импортированного функционала контекста функцию для переключения состояния
  return (
    <>
      <h1>Hello Context</h1>
      <button className="btn btn-success" onClick={toggle}>
        Show alert
      </button>
    </>
  );
}
```

```js
import { useAlert } from "./Alert/AlertContext";

//компоненте Alert
export default function Alert() {
  const alert = useAlert(); //достаем из импортированного функционала контекста переменную состояния и метод для переключения этого состояния
  if (!alert.visible) {
    return null;
  }
  return (
    <div className="alert alert-danger" onClick={alert.toggle}>
      Important message
    </div>
  );
}
```

## Композиция против контекста

Избыточная передача пропсов

```jsx
<Page user={user} avatarSize={avatarSize} />
// ... который рендерит ...
<PageLayout user={user} avatarSize={avatarSize} />
// ... который рендерит ...
<NavigationBar user={user} avatarSize={avatarSize} />
// ... который рендерит ...
<Link href={user.permalink}> //нужный пропс
  <Avatar user={user} size={avatarSize} />
</Link>
// Решение с помощью передачи компонента вниз
function Page(props) {
    const user = props.user;
    const userLink = (//определяем переменную userLink как компонент
      <Link href={user.permalink}>
        <Avatar user={user} size={props.avatarSize} />
      </Link>
    );
    return <PageLayout userLink={userLink} />;
  }// Теперь, это выглядит так:
  <Page user={user} avatarSize={avatarSize}/>
  // ... который рендерит ...
  <PageLayout userLink={...} />
  // ... который рендерит ...
  <NavigationBar userLink={...} />
  // ... который рендерит ...
  {props.userLink}

```

решение с помощью инверсии управления

```jsx
function Page(props) {
  const user = props.user;
  const content = <Feed user={user} />;
  const topBar = (
    <NavigationBar>
      <Link href={user.permalink}>
        <Avatar user={user} size={props.avatarSize} />
      </Link>
    </NavigationBar>
  );
  return <PageLayout topBar={topBar} content={content} />;
}
```

![context](/assets/react/react-context.png)

## BP. Множественный контекст

Можно создать reducer для состояния и создать два контекста, в один передавать состояния, в другой функцию reducer. Так как передать в value в Provider можно только один аргумент, если мы передадим объект, то этот объект будет вызывать новы рендер.

```tsx
import { ReactNode, createContext, useReducer } from "react";
export const ItemContext = createContext({});
export const ActionContext = createContext({});
const reducer = () => {};
const getInitialState = () => ({});
const ItemsProvider = ({ children }: { children: ReactNode }) => {
  const [items, dispatch] = useReducer(reducer, getInitialState());
  return (
    <ActionContext.Provider value={dispatch}>
      <ItemContext.Provider value={items}>{children}</ItemContext.Provider>
    </ActionContext.Provider>
  );
};
```

## Bp. асинхронная загрузка библиотек

```tsx
import {} from "react";
type SpringType = typeof import("@react-spring/web"); //импорт типов
type GestureType = typeof import("@use-gesture/react"); //импорт типов
interface AnimationContextPayload {
  Gesture?: GestureType;
  Spring?: SpringType;
  isLoaded?: boolean;
} // Обе библиотеки зависят друг от друга

//проброс через контекст
const AnimationContext = createContext<AnimationContextPayload>({});
// Асинхронная загрузку компонентов
const getAsyncAnimationModules = async () => {
  return Promise.all([
    import("@react-spring/web"),
    import("@use-gesture/react"),
  ]);
};

export const useAnimationLibs = () => {
  return useContext(AnimationContext) as Required<AnimationContextPayload>;
};
```

```tsx
export const AnimationProvider = ({ children }: { children: ReactNode }) => {
  const SpringRef = useRef<SpringType>();
  const GestureRef = useRef<GestureType>();
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    getAsyncAnimationModules().then(([Spring, Gesture]) => {
      SpringRef.current = Spring;
      GestureRef.current = Gesture;
      setIsLoaded(true);
    });
  }, []);

  const value = useMemo(
    () => ({
      Gesture: GestureRef.current,
      Spring: SpringRef.current,
      isLoaded,
    }),
    [isLoaded]
  );

  return (
    <AnimationContext.Provider value={value}>
      {children}
    </AnimationContext.Provider>
  );
};
```
