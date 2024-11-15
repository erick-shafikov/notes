Stack - навигатор, который предполагает заголовок на скрине
Tab - навигатор с меню в нижней части экрана (с помощью вложенных можно убирать)
Drawer - выезжающая панель навигации

# Компоненты

## Drawer

options для Drawer.Screen

```js
const options = {
  title?: string; //Отображаемое имя
  lazy?: boolean; //активация ленивой загрузки
  // пропсы для заголовка
  header?: (props: DrawerHeaderProps) => React.ReactNode; //полностью заменить компонент заголовка
  headerShown?: boolean;
  // пропсы для элемента меню
  drawerLabel?: string | ((props: { color: string; focused: boolean }) => React.ReactNode); //компонент бокового меню
  drawerIcon?: (props: {color: string; size: number; focused: boolean; }) => React.ReactNode; //добавит иконку у элементу бокового меню
  drawerActiveTintColor?: string; //цвет шрифта активного роута
  drawerActiveBackgroundColor?: string; //цвет background активного роута
  drawerInactiveTintColor?: string; //цвет шрифта неактивного роута
  drawerInactiveBackgroundColor?: string; //цвет background неактивного роута
  drawerAllowFontScaling?: boolean;
  drawerItemStyle?: StyleProp<ViewStyle>; //стилизация шрифта
  drawerLabelStyle?: StyleProp<TextStyle>; //стилизация контейнера
  drawerContentContainerStyle?: StyleProp<ViewStyle>; //стилизация всего контейнера
  drawerContentStyle?: StyleProp<ViewStyle>; //стилизация всего drawer
  drawerStyle?: StyleProp<ViewStyle>; //стилизация всего drawer
  drawerPosition?: 'left' | 'right';
  drawerType?: 'front' | 'back' | 'slide' | 'permanent'; // Тип drawer `front` - по умолчанию, back срабатывает на свайп, slide = front + back, permanent- как боковое меню
  drawerHideStatusBarOnOpen?: boolean;
  drawerStatusBarAnimation?: 'slide' | 'none' | 'fade';
  overlayColor?: string; //цвет overlayColor
  overlayAccessibilityLabel?: string;
  sceneContainerStyle?: StyleProp<ViewStyle>;
  gestureHandlerProps?: PanGestureHandlerProperties;
  swipeEnabled?: boolean; //true
  swipeEdgeWidth?: number; //как далеко будет реагировать на свайп
  swipeMinDistance?: number;
  keyboardDismissMode?: 'on-drag' | 'none';
  unmountOnBlur?: boolean;
  freezeOnBlur?: boolean; //true
};
```

options для header

```js
 const headerOptions = {
  headerTitle?: string | ((props: HeaderTitleProps) => React.ReactNode);
  headerTitleAlign?: 'left' | 'center'; //расположение элементов
  headerTitleStyle?: Animated.WithAnimatedValue<StyleProp<TextStyle>>; //стилизация шрифта
  headerTitleContainerStyle?: Animated.WithAnimatedValue<StyleProp<ViewStyle>>; //стилизация контейнера
  headerTitleAllowFontScaling?: boolean;
  headerLeft?: (props: { tintColor?: string; pressColor?: string; pressOpacity?: number; labelVisible?: boolean; }) => React.ReactNode; //элемент иконка в виде трех полосок
  headerLeftLabelVisible?: boolean;
  headerLeftContainerStyle?: Animated.WithAnimatedValue<StyleProp<ViewStyle>>;
  headerRight?: (props: { tintColor?: string; pressColor?: string; pressOpacity?: number; }) => React.ReactNode;
  headerRightContainerStyle?: Animated.WithAnimatedValue<StyleProp<ViewStyle>>;
  headerPressColor?: string;
  headerPressOpacity?: number; //прозрачность при нажатии
  headerTintColor?: string; //цвет
  headerBackground?: (props: {style: Animated.WithAnimatedValue<StyleProp<ViewStyle>>; }) => React.ReactNode; // бэкграунд для заголовка
  headerBackgroundContainerStyle?: Animated.WithAnimatedValue<StyleProp<ViewStyle>>;
  headerTransparent?: boolean; //false
  headerStyle?: Animated.WithAnimatedValue<StyleProp<ViewStyle>>;
  headerShadowVisible?: boolean;
  headerStatusBarHeight?: Number;
};
```

## Tab

props для tab.navigator. Так же различают Material Bottom Tabs и Material Top Tabs

```js
const tabNavigatorProps = {
  id: String,
  initialRouteName: String, //экран, который первый на рендер
  screenOptions: Object, //Объект screenOptions Tab.Navigator или options Tab.Screen см ниже (*)
  backBehavior: "initialRoute" | "firstRoute" | "history" | "order" | "none", //поведение кнопки назад
  detachInactiveScreens: true, //демонтирует неактивные экраны
  sceneContainerStyle: Object, //Объект стилей
  tabBar: ({ state, descriptors, navigation }) => Element, //функция которая возвращает React элемент, который будет являться компонентом меню
};
```

Объект screenOptions Tab.Navigator или options Tab.Screen (\*)

```js
const options = {
  title: String, // может быть использование в headerTitle или tabBarLabel
  tabBarLabel: String | ({ focused: Boolean, color: String }) => ReactNode,
  tabBarShowLabel: Boolean,
  tabBarIconStyle: Object, //Объект стилей для иконки
  tabBarBadge: String | Number, //бейдж над иконкой
  tabBarBadgeStyle: Object, //Объект стилей для бейджа
  tabBarAccessibilityLabel: String,
  tabBarTestID: String | Number,
  tabBarButton: Function,//возвращает React элемент
  // компонент и стили для иконки
  tabBarActiveTintColor: String, //цвет дял активной опции меню навигации
  tabBarInactiveTintColor: String,
  tabBarActiveBackgroundColor: String,
  tabBarInactiveBackgroundColor: String,

  tabBarHideOnKeyboard: false,
  tabBarItemStyle: Object, //стиль для всего меню
  tabBarStyle: Object,
  tabBarBackground: Function //возвращает React элемент
  lazy: true,
  unmountOnBlur: false,
  freezeOnBlur: false,

  header: ({navigation, route, options, layout }) => ReactElement
};
```

Параметр options для коллбека header

```js
const options = {
  headerStyle: Object, //стиль для заголовка
  headerShown: true,
};
```

Событийная модель

```js
React.useEffect(() => {
  const unsubscribe = navigation.addListener("tabPress", (e) => {
    // Prevent default behavior
    e.preventDefault();

    // Do something manually
    // ...
  });

  return unsubscribe;
}, [navigation]);

const EventObject = "tabLongPress" | "tabPress";
```

## BottomTabs

Навигация с помощью табов

```js
import { createNativeStackNavigator } from "@react-navigation/native-stack";

<BottomTabs.Navigator
  // общая настройка для всех оборачиваемых компонентов может быть как объектом, так и функцией, которая принимает параметры navigation, route
  screenOptions={({ navigation }) => ({
    cardOverlay: React.Element, //элемент для наложения компонента поверх
    cardShadowEnabled: true, //тени при переходе
    cardOverlayEnabled: true, //IOS === false если используем cardOverlay
    cardStyle: {}, //объект стилей дял карточки
    headerStyle: {
      //стилизация заголовка
      backgroundColor: GlobalStyles.colors.primary500,
    },
    headerTintColor: "white", //стилизация текста заголовка
    tapBarStyle: { backgroundColor: GlobalStyles.colors.primary500 },
    tapBarActiveTintColor: GlobalStyles.colors.accent500,
    headerRight: (
      { tintColor } //компонент правой части заголовка tintColor | pressColor | pressOpacity
    ) => (
      <IconButton
        icon="add"
        size={24}
        color={tintColor}
        onPress={() => {
          navigation.navigate("ManageExpense");
        }}
      />
    ),
    presentation: "card" | "modal" | "transparentModal",
    title: "title", // headerTitle
  })}
  // изначальный роут
  initialRouteName=""
  //Для оптимизации
  detachInactiveScreens={true}
>
  <BottomTabs.Screen />
</BottomTabs.Navigator>;
```

Полный список options для Stack.Screen

```js
export type NativeStackNavigationOptions = {
  title?: string, //тоже самое что и title
  header?: (props: NativeStackHeaderProps) => React.ReactNode, // кнопки слева и справа
  headerLeft?: (props: HeaderBackButtonProps) => React.ReactNode, //дял компонента с левой стороны HeaderBackButtonProps={tintColor: string, canGoBack: boolean; }
  headerRight?: (props: HeaderButtonProps) => React.ReactNode, //дял компонента с правой стороны HeaderBackButtonProps & {title?: string;}
  headerTitle?:  // проп для отображения заголовка
    | string
    | ((props: { children: string, tintColor?: string }) => React.ReactNode),
  // кнопки назад
  headerBackVisible?: boolean, // отображение кнопки назад
  headerBackTitle?: string, // текст
  headerBackTitleVisible?: boolean, // отображение текста
  headerBackTitleStyle?: StyleProp<{
    fontFamily?: string, // стиль шрифта
    fontSize?: number,
  }>,
  headerShown?: boolean, // будет ли изображаться заголовок
  headerBackImageSource?: ImageSourcePropType,
  //
  headerLargeStyle?: StyleProp<{
    backgroundColor?: string,
  }>,
  headerLargeTitle?: boolean,
  headerLargeTitleShadowVisible?: boolean,
  headerLargeTitleStyle?: StyleProp<{
    fontFamily?: string,
    fontSize?: number,
    fontWeight?: string,
    color?: string,
  }>,

  headerStyle?: StyleProp<{
    backgroundColor?: string,
  }>,
  headerShadowVisible?: boolean,
  headerTransparent?: boolean,
  headerBlurEffect?: ScreenStackHeaderConfigProps["blurEffect"],
  headerTintColor?: string,
  headerBackground?: () => React.ReactNode,
  headerTitleAlign?: "left" | "center",
  headerTitleStyle?: StyleProp<
    Pick<TextStyle, "fontFamily" | "fontSize" | "fontWeight"> & {
      color?: string,
    }
  >,
  headerSearchBarOptions?: SearchBarProps,
  headerBackButtonMenuEnabled?: boolean,
  autoHideHomeIndicator?: boolean,
  navigationBarColor?: string,
  navigationBarHidden?: boolean,
  statusBarAnimation?: ScreenProps["statusBarAnimation"],
  statusBarColor?: string,
  statusBarHidden?: boolean,
  statusBarStyle?: ScreenProps["statusBarStyle"],
  statusBarTranslucent?: boolean,
  gestureDirection?: ScreenProps["swipeDirection"],
  contentStyle?: StyleProp<ViewStyle>,
  customAnimationOnGesture?: boolean,
  fullScreenGestureEnabled?: boolean,
  gestureEnabled?: boolean,
  animationTypeForReplace?: ScreenProps["replaceAnimation"],
  animation?: ScreenProps["stackAnimation"],
  animationDuration?: number,
  presentation?: Exclude<ScreenProps["stackPresentation"], "push"> | "card",
  orientation?: ScreenProps["screenOrientation"],
  freezeOnBlur?: boolean,
};
```

Компонент отдельного скрина

```js
<BottomTabs.Screen
  //путь до компонента
  name="RecentExpenses"
  //компонент
  component={RecentExpenses}
  //свойство опций
  options={{
    title: "Recent Expenses",
    tabBarLabel: "Recent",
    tabBarIcon: ({ color, size }) => {
      return <Ionicons name="hourglass" size={size} color={color} />;
    },
  }}
  // значения по умолчанию дял route.params
  initialParams={{ itemId: 42 }}
/>
```

## NavigationContainer

Позволяет управлять состоянием навигатора

```js
import {
  NavigationContainer,
  useNavigationContainerRef,
} from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

const Stack = createNativeStackNavigator();

export default function App() {
  const navigationRef = useNavigationContainerRef();

  return (
    <NavigationContainer ref={navigationRef}>
      <Stack.Navigator>{/* ... */}</Stack.Navigator>
    </NavigationContainer>
  );
}
```

методы на ссылке

```js
navigationRef.navigate(name, params);
navigationRef.isReady;
navigationRef.getRootState();
navigationRef.getCurrentRoute();
navigationRef.getCurrentOptions();
navigationRef.addListener("state" | "options", (e) => {});
```

пропсы

```js
const NavigationContainerProps = {
  initialState: Object,
  onStateChange: (state) => {},
  onReady: () => {},
  onUnhandledAction: Function,
  linking: {
    prefixes: String[],
    config: {},
    enabled: true,
    getInitialURL: String,
    subscribe: Function,
    getStateFromPath: (path, config) => {},
    getPathFromState: ()=>{}
  },
  fallback: React.Element,
  documentTitle={
    enabled: true,
  }
  theme: Object
};
```

## ServerContainer

Позволяет работать с SSR

## Group

Позволяет роуты разделить на группы для отдельной конфигурации

```js
<Stack.Navigator>
  <Stack.Group
    screenOptions={{ headerStyle: { backgroundColor: "papayawhip" } }}
  >
    <Stack.Screen name="Home" component={HomeScreen} />
    <Stack.Screen name="Profile" component={ProfileScreen} />
  </Stack.Group>
  <Stack.Group screenOptions={{ presentation: "modal" }}>
    <Stack.Screen name="Search" component={SearchScreen} />
    <Stack.Screen name="Share" component={ShareScreen} />
  </Stack.Group>
</Stack.Navigator>
```

## Screen

Отдельный компонент для работы с отдельным экраном, пропсы

```js
const screenProps = {
  name: String,
  options: Object,
  initialParams: Object,
  getId: ({ params }) => String,
  component: Function, //функция которая возвращает React.Component,
  children: Function, //функция которая возвращает React.Component,
  navigationKey: String,
  listeners: ({ navigation, route }) => {
    tabPress: (e) => {
      e.preventDefault();

      navigation.navigate('AnotherPlace');
    },
  },
};
```

<!--Prop navigation и route в компоненте ------------------------------------------------------------------------------------------------------>

# Prop navigation и route в компоненте

## Prop navigation в компоненте

Компонент обернутый в контекст навигатора имеет доступ к объекту navigation

### navigation.navigate

Позволяет осуществить переход на другой скрин с возможностью передачи контекста

```js
//компонент из которого происходит переход. Использовать контекст можно в компоненте с помощью свойства route
const WrappedComponent = ({ navigation }) => {
  const handleRedirect = () => {
    navigation.navigate("SomeWrappedComponent", {
      // передача параметров
      someData: data,
      // передача параметров для вложенных навигаторов, если в SomeWrappedComponent вложен Settings, то переход осуществится на Settings
      screen: "Settings",
    });
  };

  // при большой вложенности
  navigation.navigate("Root", {
    screen: "Settings",
    params: {
      screen: "Sound",
      params: {
        screen: "Media",
      },
    },
  });

  const handleGoBack = () => {
    // на одну страницу назад
    navigation.goBack();
    // при navigate на туже страницу перенаправит в отличие от navigate
    navigation.push();
    //на самый верхний уровень
    navigation.popToTop();
  };
};
```

### navigation.setOptions

Позволяет настроить навигационные параметры

```js
//установит заголовок
navigation.setOptions({
  title: "title",
});
```

### navigation.isFocused

Позволяет определить находится ли в фокусе скрин

### navigation.addListener:ƒ (type, callback)

type, доступные события:

- focus
- state
- blur
- beforeRemove

Позволяет повесить слушателя на события перехода

```js
//пример с предотвращением перехода
function EditText({ navigation }) {
  const [text, setText] = React.useState("");
  // флаг доступа перехода
  const hasUnsavedChanges = Boolean(text);

  React.useEffect(
    () =>
      navigation.addListener("beforeRemove", (e) => {
        if (!hasUnsavedChanges) {
          return; //если переход запрещен
        }
        e.preventDefault(); //если разрешен предотвращаем нативный переход
        Alert.alert(
          "Discard changes?",
          "You have unsaved changes. Are you sure to discard them and leave the screen?",
          [
            { text: "Don't leave", style: "cancel", onPress: () => {} },
            {
              text: "Discard",
              style: "destructive",
              onPress: () => navigation.dispatch(e.data.action), //выполняем действие по переходу
            },
          ]
        );
      }),
    [navigation, hasUnsavedChanges]
  );

  return (
    <TextInput
      value={text}
      placeholder="Type something…"
      onChangeText={setText}
    />
  );
}
```

### actions

каждый action имеет следующие поля type, payload {name, params}, source ,target

- navigate - name, key, params
- reset
- goBack
- setParams

```js
import { CommonActions } from "@react-navigation/native";

navigation.dispatch(
  CommonActions.navigate({
    name: "Profile",
    params: {
      user: "jane",
    },
  })
);
```

**_StackActions_** имеет действия replace, push

**_DrawerActions_** имеет openDrawer, closeDrawer, toggleDrawer, jumpTo

**_TabActions_** имеет jumpTo

**navigation.canGoBack**:ƒ ()

**navigation.dispatch**:ƒ (thunk)

**navigation.getId**:ƒ ()

**navigation.getParent**:ƒ (id)

**navigation.getState**:ƒ ()

```js
const state = {
  type: "stack",
  key: "stack-1",
  routeNames: ["Home", "Profile", "Settings"],
  routes: [
    { key: "home-1", name: "Home", params: { sortBy: "latest" } },
    { key: "settings-1", name: "Settings" },
  ],
  index: 1,
  stale: false,
};
```

**navigation.removeListener**:ƒ (type, callback)
**navigation.replace**:ƒ ()
**navigation.reset**:ƒ ()
**navigation.setParams**: ƒ ()

## Prop route в компоненте

```js
//компонент в который происходит переход, можно получить контекст, который был передан в параметрах
const SomeWrappedComponent = ({ route }) => {
  const someParams = route.params.someParams;

  // полное описание объекта route
  /* const route = {
  key: SecondScreen-JpuFNrUoN4tPp7DSt2hu_
  name: "SecondScreen"
  params:
  path: undefined
} */
};
```

<!--Хуки----------------------------------------------------------------------------------------------------------------------------------------->

# Хуки

## useFocusEffect

хук который срабатывает при фокусе на скрин

```js
import { useFocusEffect } from "@react-navigation/native";

function Profile({ userId }) {
  const [user, setUser] = React.useState(null);

  useFocusEffect(
    React.useCallback(() => {
      const unsubscribe = API.subscribe(userId, (user) => setUser(data));

      return () => unsubscribe();
    }, [userId])
  );

  return <ProfileContent user={user} />;
}
//альтернативный подход с прослушиванием событий
React.useEffect(() => {
  const unsubscribe = navigation.addListener("blur", () => {
    // Do something when the screen blurs
  });

  return unsubscribe;
}, [navigation]);
```

## useIsFocused

Позволяет узнать находится ли в фокуса скрин

```js
import { useIsFocused } from "@react-navigation/native";

function Profile() {
  const isFocused = useIsFocused();

  return <Text>{isFocused ? "focused" : "unfocused"}</Text>;
}
```

## useLinkTo

Позволяет работать с linking options

```js
import { useLinkTo } from "@react-navigation/native";

function Home() {
  const linkTo = useLinkTo();

  return (
    <Button onPress={() => linkTo("/profile/jane")}>
      Go to Jane's profile
    </Button>
  );
}
```

## useLinkBuilder

позволяет создавать пути при использовании linkBuilder'а

## useLinkProps

Позволяет создавать кастомные компоненты навигации

```js
import { useLinkTo } from "@react-navigation/native";

function Home() {
  const linkTo = useLinkTo();

  return (
    <Button onPress={() => linkTo("/profile/jane")}>
      Go to Jane's profile
    </Button>
  );
}
```

## useNavigation и useRoute

возвращает объект navigation в объектах большей вложенности

```js
import { useNavigation, useRoute } from "@react-navigation/native";

const ComponentThaUseUseNavigation = () => {
  const navigation = useNavigation();
  // route.params содержит контекст
  const route = useRoute();

  const handleClick = () =>
    navigation.navigate("someWhere", { param: "someParams" });
};
```

## useNavigationState

позволяет получить состояние роутера

```js
function Profile() {
  const routesLength = useNavigationState((state) => state.routes.length);

  return <Text>Number of routes: {routesLength}</Text>;
}
```

## useRoute

позволяет получить объект route без свойства route

```js
import { useRoute } from "@react-navigation/native";

function MyText() {
  const route = useRoute();

  const { someData } = route.params;
}
```

## useSafeAreaInsets

Хук позволяет предотвратить перекрытие контента частями телефона

```js
import { useSafeAreaInsets } from "react-native-safe-area-context";

function Demo() {
  // хук возвращает значения для отступов
  const insets = useSafeAreaInsets();

  return (
    <View
      // применяем для контейнеров в стилях
      style={{
        paddingTop: insets.top,
        paddingBottom: insets.bottom,
      }}
    ></View>
  );
}
```

## useScrollToTop

позволяет прокручивать к верху компонента

```js
import { useScrollToTop } from "@react-navigation/native";

function Albums() {
  const ref = React.useRef(null);

  useScrollToTop(ref);

  return <ScrollView ref={ref}>{/* content */}</ScrollView>;
}
```

## useTheme

```js
//в корневом компоненте
import { NavigationContainer, DefaultTheme } from "@react-navigation/native";

const MyTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: "rgb(255, 45, 85)",
  },
};

export default function App() {
  return (
    //передаем тему
    <NavigationContainer theme={MyTheme}>{/* content */}</NavigationContainer>
  );
}
//в компоненте
import { useTheme } from "@react-navigation/native";
function MyButton() {
  const { colors } = useTheme();

  return (
    <TouchableOpacity style={{ backgroundColor: colors.card }}>
      <Text style={{ color: colors.text }}>Button!</Text>
    </TouchableOpacity>
  );
}
```

# Вспомогательные функции

## getFocusedRouteNameFromRoute

Функция позволяет получить имя активного роута, только внутри своего контекста роутинга. Функция getFocusedRouteNameFromRoute позволяет получить путь вложенных контекстов

```js
import { getFocusedRouteNameFromRoute } from "@react-navigation/native";
//вспомогательная функция
function getHeaderTitle(route) {
  const routeName = getFocusedRouteNameFromRoute(route) ?? "Feed";
  switch (routeName) {
    case "Feed":
      return "News feed";
    case "Profile":
      return "My profile";
    case "Account":
      return "My account";
  }
}
//использование

function FeedStackScreen() {
  return (
    <FeedStack.Navigator>
      <FeedStack.Screen
        name="Feed"
        component={FeedScreen}
        options={({ route }) => ({
          headerTitle: getHeaderTitle(route),
        })}
      />
      {/* other screens */}
    </FeedStack.Navigator>
  );
}

//вложенный в HomeTabs
const ProfileStack = createNativeStackNavigator();
function ProfileStackScreen() {
  return (
    <ProfileStack.Navigator>
      <ProfileStack.Screen name="Profile" component={ProfileScreen} />
      {/* other screens */}
    </ProfileStack.Navigator>
  );
}

//вложенный в App
const Tab = createBottomTabNavigator();
function HomeTabs() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="Feed" component={FeedStackScreen} />
      <Tab.Screen name="Profile" component={ProfileStackScreen} />
    </Tab.Navigator>
  );
}

const RootStack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <RootStack.Navigator>
        <RootStack.Screen name="Home" component={HomeTabs} />
        <RootStack.Screen name="Settings" component={SettingsScreen} />
      </RootStack.Navigator>
    </NavigationContainer>
  );
}
```

# Deep linking

** Нужно если:**
Нужно перенаправить пользователя, при запуске приложения на какой-либо роут при включенном или выключенном состоянии

```js
//в app.json

{
  "expo": {
    "scheme": "mychat"
  }
}

//npx expo install expo-linking

//активация deep linking
import { NavigationContainer } from "@react-navigation/native";
//создать объект linking
const linking = {
  prefixes: [],
  config: {},
};

function App() {
  return (
    // передать его в NavigationContainer
    <NavigationContainer linking={linking} fallback={<Text>Loading...</Text>}>
      {/* content */}
    </NavigationContainer>
  );
}
```

## Объект linking

```js
const linking = {
  //корневой путь
  prefixes: ["mychat://", "https://mychat.com", "https://*.mychat.com"],
  //функция которая позволяет фильтровать входящие ссылки
  filter: (url) => !url.includes("+expo-auth-session"),
};
```

**config**

При передачи параметров

```js
// /user/wojciech/settings
// { id: 'user-wojciech' section: 'settings' }
//конфиг
const config = {
  screens: {
    // для Profile
    Profile: {
      // знак ? на конце строки делает его опциональным
      path: "user/:id/:section?",
      parse: {
        id: (id) => `user-${id}`,
      },
      stringify: {
        id: (id) => id.replace(/^user-/, ""),
      },
    },
  },
  getStateFromPath: (path, options) => {
    // Return a state object here
    // You can also reuse the default logic by importing `getStateFromPath` from `@react-navigation/native`
  },
  getPathFromState(state, config) {
    // Return a path string here
    // You can also reuse the default logic by importing `getPathFromState` from `@react-navigation/native`
  },
};
//состояние
const state = {
  routes: [
    {
      name: "Profile",
      params: { id: "user-wojciech", section: "settings" },
    },
  ],
};
```

## Вложенные роуты

- не передаются params, использовать контекст

```js
//контейнер для вложенных табов
const TabContainer = () => (
  <Tab.Navigator>
    <Tab.Screen name="TabFirst" component={TabFirstScreen} />
    <Tab.Screen name="TabSecond" component={TabSecondScreen} />
  </Tab.Navigator>
);
//корневой компонент
export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="FirstScreen" component={FirstScreen} />
        {/* должен быть обернут во внешний роут */}
        <Stack.Screen name="Tabs" component={TabContainer} />
        <Stack.Screen name="SecondScreen" component={SecondScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const SecondScreen = ({ navigation }) => {
  return (
    <View>
      <Button
        title="go to 1st tab"
        onPress={() => {
          //переход во вложенный таб указываем происходит с помощью внешнего роута
          navigation.navigate("Tabs", { screen: "TabFirst" });
        }}
      />
    </View>
  );
};
```

```js
function Home() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="Profile" component={Profile} />
      <Tab.Screen name="Feed" component={Feed} />
    </Tab.Navigator>
  );
}

function App() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={Home} />
      <Stack.Screen name="Settings" component={Settings} />
    </Stack.Navigator>
  );
}
//конфиг
const config = {
  screens: {
    Home: {
      screens: {
        Profile: "users/:id",
      },
    },
  },
};
//состояние
const state = {
  routes: [
    {
      name: "Home",
      state: {
        routes: [
          {
            name: "Profile",
            params: { id: "jane" },
          },
        ],
      },
    },
  ],
};
```

Конфиг для 404

```js
const config = {
  screens: {
    Home: {
      initialRouteName: "Feed",
      screens: {
        Profile: "users/:id",
        Settings: "settings",
      },
    },
    NotFound: "*",
  },
};

const state = {
  routes: [{ name: "NotFound" }],
};
```
