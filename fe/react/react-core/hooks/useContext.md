# useContext

```tsx
//РєРѕРјРїРѕРЅРµРЅС‚ РѕС‚РІРµС‡Р°СЋС‰РёР№ Р·Р° РєРѕРЅС‚РµРєСЃС‚
import { useContext, createContext } from "react";

const AlertContext = createContext(null); //СЃРѕР·РґР°РµРј РѕР±СЉРµРєС‚ РєРѕРЅС‚РµРєСЃС‚Р°

export const AlertProvider = ({ children }: { children: React.ReactNode }): JSX.Element => {
  //РѕР±РµСЂС‚РєР° AlertProvider РґР»СЏ РІСЃРµРіРѕ РїСЂРёР»РѕР¶РµРЅРёСЏ
  const [alert, setAlert] = useState<boolean>(false); //Р»РѕРіРёРєР°
  const toggle = () => setAlert(!alert);

  return (
    //c R19 РјРѕР¶РЅРѕ РїСЂРѕСЃС‚Рѕ <AlertContext ...>
    <AlertContext.Provider value={{ visible: alert, toggle }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlert = () => useContext(AlertContext); //С„СѓРЅРєС†РёСЏ РґР»СЏ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёСЏ РєРѕРЅС‚РµРєСЃС‚Р° РІ РєРѕРјРїРѕРЅРµРЅС‚Р°С…
```

```tsx
import Alert from "./Alert";
import { AlertProvider } from "./Alert/AlertContext";
import Main from "./Main";
//РіР»Р°РІРЅРѕРµ РїСЂРёР»РѕР¶РµРЅРёРµ СЂРµРЅРµРґРµСЂРёС‚ РґРІР° РєРѕРјРїРѕРЅРµРЅС‚Рµ
function UseContextHook(): JSX.Element {
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

```tsx
//РєРѕРјРїРѕРЅРµРЅС‚Рµ Main
import { useAlert } from "./Alert/AlertContext";

export default function Main(): JSX.Element {
  const { toggle } = useAlert(); //РґРѕСЃС‚Р°РµРј РёР· РёРјРїРѕСЂС‚РёСЂРѕРІР°РЅРЅРѕРіРѕ С„СѓРЅРєС†РёРѕРЅР°Р»Р° РєРѕРЅС‚РµРєСЃС‚Р° С„СѓРЅРєС†РёСЋ РґР»СЏ РїРµСЂРµРєР»СЋС‡РµРЅРёСЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ
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

```tsx
import { useAlert } from "./Alert/AlertContext";

//РєРѕРјРїРѕРЅРµРЅС‚Рµ Alert
export default function Alert(): JSX.Element | null {
  const alert = useAlert(); //РґРѕСЃС‚Р°РµРј РёР· РёРјРїРѕСЂС‚РёСЂРѕРІР°РЅРЅРѕРіРѕ С„СѓРЅРєС†РёРѕРЅР°Р»Р° РєРѕРЅС‚РµРєСЃС‚Р° РїРµСЂРµРјРµРЅРЅСѓСЋ СЃРѕСЃС‚РѕСЏРЅРёСЏ Рё РјРµС‚РѕРґ РґР»СЏ РїРµСЂРµРєР»СЋС‡РµРЅРёСЏ СЌС‚РѕРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ
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

## РљРѕРјРїРѕР·РёС†РёСЏ РїСЂРѕС‚РёРІ РєРѕРЅС‚РµРєСЃС‚Р°

РР·Р±С‹С‚РѕС‡РЅР°СЏ РїРµСЂРµРґР°С‡Р° РїСЂРѕРїСЃРѕРІ

```tsx
<Page user={user} avatarSize={avatarSize} />
// ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
<PageLayout user={user} avatarSize={avatarSize} />
// ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
<NavigationBar user={user} avatarSize={avatarSize} />
// ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
<Link href={user.permalink}> //РЅСѓР¶РЅС‹Р№ РїСЂРѕРїСЃ
В  <Avatar user={user} size={avatarSize} />
</Link>
// Р РµС€РµРЅРёРµ СЃ РїРѕРјРѕС‰СЊСЋ РїРµСЂРµРґР°С‡Рё РєРѕРјРїРѕРЅРµРЅС‚Р° РІРЅРёР·
function Page(props) {
В  В  const user = props.user;
В  В  const userLink = (//РѕРїСЂРµРґРµР»СЏРµРј РїРµСЂРµРјРµРЅРЅСѓСЋ userLink РєР°Рє РєРѕРјРїРѕРЅРµРЅС‚
В  В  В  <Link href={user.permalink}>
В  В  В  В  <Avatar user={user} size={props.avatarSize} />
В  В  В  </Link>
В  В  );
В  В  return <PageLayout userLink={userLink} />;
В  }// РўРµРїРµСЂСЊ, СЌС‚Рѕ РІС‹РіР»СЏРґРёС‚ С‚Р°Рє:
В  <Page user={user} avatarSize={avatarSize}/>
В  // ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
В  <PageLayout userLink={...} />
В  // ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
В  <NavigationBar userLink={...} />
В  // ... РєРѕС‚РѕСЂС‹Р№ СЂРµРЅРґРµСЂРёС‚ ...
В  {props.userLink}

```

СЂРµС€РµРЅРёРµ СЃ РїРѕРјРѕС‰СЊСЋ РёРЅРІРµСЂСЃРёРё СѓРїСЂР°РІР»РµРЅРёСЏ

```tsx
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

## BP. РњРЅРѕР¶РµСЃС‚РІРµРЅРЅС‹Р№ РєРѕРЅС‚РµРєСЃС‚

РњРѕР¶РЅРѕ СЃРѕР·РґР°С‚СЊ reducer РґР»СЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ Рё СЃРѕР·РґР°С‚СЊ РґРІР° РєРѕРЅС‚РµРєСЃС‚Р°, РІ РѕРґРёРЅ РїРµСЂРµРґР°РІР°С‚СЊ СЃРѕСЃС‚РѕСЏРЅРёСЏ, РІ РґСЂСѓРіРѕР№ С„СѓРЅРєС†РёСЋ reducer. РўР°Рє РєР°Рє РїРµСЂРµРґР°С‚СЊ РІ value РІ Provider РјРѕР¶РЅРѕ С‚РѕР»СЊРєРѕ РѕРґРёРЅ Р°СЂРіСѓРјРµРЅС‚, РµСЃР»Рё РјС‹ РїРµСЂРµРґР°РґРёРј РѕР±СЉРµРєС‚, С‚Рѕ СЌС‚РѕС‚ РѕР±СЉРµРєС‚ Р±СѓРґРµС‚ РІС‹Р·С‹РІР°С‚СЊ РЅРѕРІС‹ СЂРµРЅРґРµСЂ.

```tsx
import React, {
  useMemo,
  createContext,
  useContext,
  useState,
  useCallback,
} from "react";

// ----------------------------------------------------------------------
// СЃРѕР·РґР°РµРј РєРѕРЅС‚РµРєСЃС‚ РґР»СЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ
const __Entity__CreateStateContext = createContext(null);

if (process.env.NODE_ENV !== "production") {
  __Entity__CreateStateContext.displayName = "__Entity__CreateStateContext";
}
// СЃРѕР·РґР°РµРј РєРѕРЅС‚РµРєСЃС‚ РґР»СЏ СЃРµС‚С‚РµСЂРѕРІ СЃРѕСЃС‚РѕСЏРЅРёСЏ
const __Entity__CreateActionsContext = createContext(null);

if (process.env.NODE_ENV !== "production") {
  __Entity__CreateActionsContext.displayName = "__Entity__CreateActionsContext";
}

// ----------------------------------------------------------------------

export const __Entity__CreateProvider = (props) => {
  const { children } = props;

  //РґР»СЏ РїСЂРёРјРµСЂР° РґРІР° СЃРѕСЃС‚РѕСЏРЅРёСЏ
  const [state_1, setState_1] = useState();
  const [state_2, setState_2] = useState();

  //РєРѕР»Р»Р±РµРєРё РЅР° РёР·РјРµРЅРµРЅРёСЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ
  const set__Entity__State_1 = useCallback((step) => {
    setState_1(step);
  }, []);

  const set__Entity__state_2 = useCallback((steps) => {
    setState_2(steps);
  }, []);

  //РјРµРјРѕРёР·РёСЂРѕРІР°РЅРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ
  const state = useMemo(() => {
    return {
      __entity__State_1: state_1,
      __entity__State_2: state_2,
    };
  }, [state_1, state_2]);
  //РјРµРјРѕРёР·РёСЂРѕРІР°РЅРЅС‹Рµ РєРѕР»Р»Р±РµРєРё
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
// С…СѓРє РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ
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

## Bp. Р°СЃРёРЅС…СЂРѕРЅРЅР°СЏ Р·Р°РіСЂСѓР·РєР° Р±РёР±Р»РёРѕС‚РµРє

```tsx
type SpringType = typeof import("@react-spring/web"); //РёРјРїРѕСЂС‚ С‚РёРїРѕРІ
type GestureType = typeof import("@use-gesture/react"); //РёРјРїРѕСЂС‚ С‚РёРїРѕРІ
interface AnimationContextPayload {
  Gesture?: GestureType;
  Spring?: SpringType;
  isLoaded?: boolean;
} // РћР±Рµ Р±РёР±Р»РёРѕС‚РµРєРё Р·Р°РІРёСЃСЏС‚ РґСЂСѓРі РѕС‚ РґСЂСѓРіР°

//РїСЂРѕР±СЂРѕСЃ С‡РµСЂРµР· РєРѕРЅС‚РµРєСЃС‚
const AnimationContext = createContext<AnimationContextPayload>({});
// РђСЃРёРЅС…СЂРѕРЅРЅР°СЏ Р·Р°РіСЂСѓР·РєСѓ РєРѕРјРїРѕРЅРµРЅС‚РѕРІ
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

## bps. С„Р°Р±СЂРёРєР° РєРѕРЅС‚РµРєСЃС‚РѕРІ

```tsx
import React from "react";

export const createContext = <T extends {}>() => {
  const Context = React.createContext<T | undefined>(undefined);

  const useContext = () => {
    const ctx = React.useContext(Context);
    if (ctx === undefined) {
      throw new Error("useContext must be used within a Provider");
    }
    return ctx;
  };

  return [useContext, Context.Provider] as const;
};
```
