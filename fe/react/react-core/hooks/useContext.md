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
    //c R19 можно просто <AlertContext ...>
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

# BPs:

## BP. Множественный контекст

Можно создать reducer для состояния и создать два контекста, в один передавать состояния, в другой функцию reducer. Так как передать в value в Provider можно только один аргумент, если мы передадим объект, то этот объект будет вызывать новы рендер.

```jsx
import React, {
  useMemo,
  createContext,
  useContext,
  useState,
  useCallback,
} from "react";

// ----------------------------------------------------------------------
// создаем контекст для состояния
const __Entity__CreateStateContext = createContext(null);

if (process.env.NODE_ENV !== "production") {
  __Entity__CreateStateContext.displayName = "__Entity__CreateStateContext";
}
// создаем контекст для сеттеров состояния
const __Entity__CreateActionsContext = createContext(null);

if (process.env.NODE_ENV !== "production") {
  __Entity__CreateActionsContext.displayName = "__Entity__CreateActionsContext";
}

// ----------------------------------------------------------------------

export const __Entity__CreateProvider = (props) => {
  const { children } = props;

  //для примера два состояния
  const [state_1, setState_1] = useState();
  const [state_2, setState_2] = useState();

  //коллбеки на изменения состояния
  const set__Entity__State_1 = useCallback((step) => {
    setState_1(step);
  }, []);

  const set__Entity__state_2 = useCallback((steps) => {
    setState_2(steps);
  }, []);

  //мемоизированное состояние
  const state = useMemo(() => {
    return {
      __entity__State_1: state_1,
      __entity__State_2: state_2,
    };
  }, [state_1, state_2]);
  //мемоизированные коллбеки
  const actions = useMemo(() => {
    return {
      set__Entity__state_1,
      set__Entity__tate_2,
    };
  }, []);

  return (
    <PayoutsCreateStateContext.Provider value={state}>
      <PayoutsCreateActionsContext.Provider value={actions}>
        {children}
      </PayoutsCreateActionsContext.Provider>
    </PayoutsCreateStateContext.Provider>
  );
};

// ----------------------------------------------------------------------
// хук для получения
export const usePayoutsCreateStateContext = () => {
  const stateContext = useContext(__Entity__CreateStateContext);

  if (!stateContext) {
    throw new Error(
      "usePayoutsCreateStateContext must be used inside a PayoutsCreateStateContext"
    );
  }

  return stateContext;
};

export const usePayoutsCreateActionsContext = () => {
  const actionsContext = useContext(__Entity__CreateActionsContext);

  if (!actionsContext) {
    throw new Error(
      "usePayoutsCreateActionsContext must be used inside a PayoutsCreateActionsContext."
    );
  }

  return actionsContext;
};
```

## Bp. асинхронная загрузка библиотек

```tsx
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
