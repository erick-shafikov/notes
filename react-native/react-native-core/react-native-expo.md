# EXPO

## Build. Работа со сборкой

- Сборку можно реализовать с помощью EAS Build или на компьютере
- с помощью --tunnel флага, можно получить ссылку по которой будет доступен проект

## Deploy. Деплой проекта

## EAS (Expo Application Services)

EAS Облачный сервис, который позволяет работать со сборкой expo

## Работа с плагинами

Есть возможность добавлять собственные плагины для работы со сборкой

```js
//кастомный плагин добавит произвольное значение в Info.plist
//my-plugin.js
const withMySDK = (config, { apiKey }) => {
  if (!config.ios) {
    config.ios = {};
  }
  if (!config.ios.infoPlist) {
    config.ios.infoPlist = {};
  }

  config.ios.infoPlist["MY_CUSTOM_NATIVE_IOS_API_KEY"] = apiKey;

  return config;
};

module.exports.withMySDK = withMySDK;

//app.config.js

const { withMySDK } = require("./my-plugin");

const config = {
  name: "my app",
};

module.exports = withMySDK(config, { apiKey: "X-XXX-XXX" });
```

Можно создавать плагины для дебага

## Тестирование приложений перед релизом

Конфигурация

```js
// store.config.json
{
  "configVersion": 0,
  "apple": {
    "info": {
      "en-US": {
        "title": "Awesome App",
        "subtitle": "Your self-made awesome app",
        "description": "The most awesome app you have ever seen",
        "keywords": ["awesome", "app"],
        "marketingUrl": "https://example.com/en/promo",
        "supportUrl": "https://example.com/en/support",
        "privacyPolicyUrl": "https://example.com/en/privacy"
      }
    }
  }
}

```

## ENV

```js
//.env
EXPO_PUBLIC_API_URL=https://staging.example.com
EXPO_PUBLIC_API_KEY=abc123
//использование

 const apiUrl = process.env.EXPO_PUBLIC_API_URL;
```

<!--Файлы
конфигурации--------------------------------------------------------------------------------------------------------------------->

# Configurations. Файлы конфигурации

## app.json

```js
{
  expo: {
    name: String;
    description: String;
    slug: String; //приложение будет доступно по expo.dev/project-owner/slug
    owner: String;
    backgroundColor: type Color,
    //splash - настройка экрана загрузки приложения
    splash: {
      image: "url/"; //url гда находится картинка

      backgroundColor: "#color"; //цветовой фон позади картинки
      resizeMode: "contain" | "cover" | "overlayed";
    }

    plugins: [
      //доступно только для SDK 50, доступно через fontFamily
      ["expo-font"],
      {
        fonts: "./assets/...",
      },
    ];
  }
}
```

<!--Навигация--------------------------------------------------------------------------------------------------------------------->

# Навигация

## Навигация. общие подходы

### Файловая структура

expo использует навигацию по файлам, все страницы должный лежать в папке app

- app (корневой каталог)
- - \_layout.tsx (компонент-обертка)
- - index.tsx (контент страницы доступный по адресу '/')
- - home.tsx (контент страницы доступный по адресу '/home')
- - Папка settings
- - - index.tsx (контент страницы доступный по адресу '/settings')
- - Папка blog
- - - [id].tsx (файл доступный по адресу "./blog/[id]") получить можно с помощью хука let {id} = useLocalSearchParams()

### Группы роутов

Позволяют сгруппировать роуты не создавая дополнительный параметр в пути

Пример 1

- app
- - \_layout.js
- - (home) //shared router
- - - index.js
- - - details.js
- - - \_layout.js

```js
// app/(home)/_layout.js
import { Stack } from "expo-router";

export default function HomeLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: {
          backgroundColor: "#f4511e",
        },
        headerTintColor: "#fff",
        headerTitleStyle: {
          fontWeight: "bold",
        },
      }}
    >
      <Stack.Screen name="index" />
      <Stack.Screen name="details" />
    </Stack>
  );
}
// app/_layout.js

import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen name="(home)" />
    </Stack>
  );
}

```

Пример 2

- app
- - (app) //shared router
- - - index.js
- - - user.js
- - (routes) //shared router
- - - route1.js
- - - route2.js

Массивный синтекс

```js
//app/(home,search)/_layout.tsx
export default function DynamicLayout({ segment }) {
  if (segment === "(search)") {
    return <SearchStack />;
  }

  return <Stack />;
}
```

<!--Навигация.Компоненты--------------------------------------------------------------------------------------------------------------------->

## Навигация. Компоненты

### Drawer

Drawer предоставляет навигационное меню, которое выезжает/заезжает

```js
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { Drawer } from "expo-router/drawer";

export default function Layout() {
  return (
    <GestureHandlerRootView _style={{ flex: 1 }}>
      <Drawer />
    </GestureHandlerRootView>
  );
}
```

в случае кастомизации

```js
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { Drawer } from "expo-router/drawer";

export default function Layout() {
  return (
    <GestureHandlerRootView _style={{ flex: 1 }}>
      <Drawer>
        <Drawer.Screen name="index" options={{}} />
        <Drawer.Screen name="user/[id]" options={{}} />
      </Drawer>
    </GestureHandlerRootView>
  );
}
```

### ErrorBoundary обработка ошибок

**использование компонента по умолчанию**

```js
//app/[...unmatched].js
export default Unmatched;

import { Unmatched } from "expo-router";
export default Unmatched;
```

**ErrorBoundary**

```js
import { View, Text } from "react-native";

//компонент ErrorBoundary
// ErrorBoundaryProps : {error: Error, retry: () => Promise<void> }
export function ErrorBoundary(props: ErrorBoundaryProps) {
  return (
    <View _style={{ flex: 1, backgroundColor: "red" }}>
      <Text>{props.error.message}</Text>
      <Text onPress={props.retry}>Try Again?</Text>
    </View>
  );
}

//страница
const function Page() {}
```

### Link

Навигация может осуществляться по средству использования компонента **Link**
Компонент ссылки, как кнопка. Link - обернутый Text компонент

```js
import { Link } from "expo-router";

export default function Page() {
  return (
    <View>
      <Link href="/about">About</Link>
      <Link href="/user/bacon">View user</Link>
      {/* проп asChild позволяет передать все пропсы первому дочернему компоненту */}
      <Link href="/other" asChild>
        <Pressable>
          <Text>Home</Text>
        </Pressable>
      </Link>
      {/* при переходя по динамическим роутам */}
      <Link
        href={{
          pathname: "/user/[id]",
          params: { id: "bacon" },
        }}
      >
        View user
      </Link>
      {/* с учетом типизации 
        <Link href="/about" />
        <Link href="/user/1" />
        <Link href={`/user/${id}`} />
        <Link href={("/user" + id) as Href} />
      */}
      <Link href={{ pathname: "/user/[id]", params: { id: 1 } }} />
    </View>
  );
}
```

При навигации можно использовать один из следующих флагов

- **navigate** перенаправит к ближайшему в стеке. Добавляет в стек новый роут, если он отличается от текущего
- **push** всегда добавляет новый роут
- **replace** меняет текущий роут на новый

что бы осуществить навигацию по принципу push или replace

```js
import { Link } from "expo-router";

export default function Page() {
  return (
    <View>
      {/* проп push осуществляет навигацию по принципу push */}
      <Link push href="/feed">
        Login
      </Link>
      {/* проп push осуществляет навигацию по принципу replace */}
      <Link replace href="/feed">
        Login
      </Link>
    </View>
  );
}
```

### Modal

Что бы какой-либо роут представить в виде модального окна можно назвать файл modal

- /app
- - \_layout.js
- - home.js
- - modal.jd

```js
// _layout.js
import { Stack } from "expo-router";
export default function Layout() {
  return (
    <Stack>
    <Stack.Screen
        name="home"
        options={{
          // Hide the header for all other routes.
          headerShown: false,
        }}
      />
      {/* поместить в stack */}
      <Stack.Screen
        name="modal"
        options={{
          // Set the presentation mode to modal for our modal route.
          presentation: "modal",
        }}
      />
    </Stack>
  );
}
//home.js
export default function Home() {
  return (
    <View _style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Text>Home Screen</Text>
      {/* ссылка по которой можно открыть модальное окно */}
      <Link href="/modal">Present modal</Link>
    </View>
  );
}

import { View } from 'react-native';
import { Link, router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
export default function Modal() {
  // навигация может произойти извне
  const isPresented = router.canGoBack();
  return (
    <View _style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      {/* для случая когда нельзя перейти назад по стеку */}
      {!isPresented && <Link href="../">Dismiss</Link>}
      {/* Native modals have dark backgrounds on iOS, set the status bar to light content. */}
      <StatusBar _style="light" />
    </View>
  );
}

```

### NotFound

```js
import { Link, Stack } from "expo-router";
import { View, StyleSheet } from "react-native";

export default function NotFoundScreen() {
  return (
    <>
      <Stack.Screen options={{ title: "Oops! This screen doesn't exist." }} />
      <View _style={styles.container}>
        <Link href="/">Go to home screen</Link>
      </View>
    </>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});
```

### Redirect

компонент, который позволяет произвести перенаправление

```js
import { Redirect } from "expo-router";

export default function Page() {
  const { user } = useAuth();

  if (!user) {
    return <Redirect href="/login" />;
  }

  return (
    <View>
      <Text>Welcome Back!</Text>
    </View>
  );
}
```

### Safe area

```js
//в layout определяем SafeAreaProvider
import { SafeAreaProvider } from "react-native-safe-area-context";

function RootLayoutNav() {
  const colorScheme = useColorScheme();

  return (
    //компонент SafeAreaProvider
    <SafeAreaProvider>
      <Stack>
        <Screen name="index" options={{ headerShown: false }} />
      </Stack>
    </SafeAreaProvider>
  );
}

//в компоненте index.js
import { StyleSheet, View, Text } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

export default function TabOneScreen() {
  //используем отступы в компоненте
  const insets = useSafeAreaInsets();

  return (
    <View _style={[styles.container, { paddingTop: insets.top }]}>
      <Text _style={styles.title}>Home page</Text>
    </View>
  );
}
```

<!-- навигационный компоненты -->

### Slot

Компонент Slot позволяет обернуть в Layout компонент, который находится в файле index, это базовый случай подключения роутинга
Он не предоставляет никаких дополнительных элементов Ui

```js
import { Slot } from "expo-router";
// Slot не принимает проп children
// export declare function Slot(props: Omit<NavigatorProps, 'children'>): JSX.Element;
export default function HomeLayout() {
  return (
    <>
      <Header />
      <Slot />
      <Footer />
    </>
  );
}

/*
При вложенных роутах
/app
- layout.js
- /home
  - _layout.js
  - index.js
*/

// app/_layout.js корневой файл определяет как Tab
import { Tabs } from 'expo-router';

export default function Layout() {
  return <Tabs />;
}

//app/home/layout.js вложенный как Stack
import { Stack } from 'expo-router';

export default function Layout() {
  return <Stack screenOptions={{}}/>;
}
```

### Stack

Основной элемент навигации, предоставляет только панель верхнего меню

- /app
- - layout.js
- - index.js
- - detail.js

```js
// _layout.js
import { Stack } from "expo-router/stack";

// Stack обернет в роутер
export default function Layout() {
  return (
    <Stack
      screenOptions={
        {
          // screenOptions из react navigation
        }
      }
    />
  );
}
```

### Stack.screen

позволяет конфигурировать каждый отдельный роут, конфигурацию можно осуществить непосредственно со страницы можно непосредственно из страницы

```js
// _layout.js
function Home() {
  return (
    // внутри компонента можно конфигурировать опции Stack
    <View>
      <Stack.Screen
        options={
          {
            // screenOptions из react navigation
          }
        }
      />
    </View>
  );
}
```

#### кастомизированный Stack

```js
//components/customStack
import { ParamListBase, StackNavigationState } from '@react-navigation/native';
import {
  createStackNavigator,
  StackNavigationEventMap,
  StackNavigationOptions,
} from '@react-navigation/stack';
import { withLayoutContext } from 'expo-router';

const { Navigator } = createStackNavigator();

export const CustomStack = withLayoutContext<
  StackNavigationOptions,
  typeof Navigator,
  StackNavigationState<ParamListBase>,
  StackNavigationEventMap
>(Navigator);
```

```js
import { CustomStack } from "components/customStack";

export default function Layout() {
  return (
    <CustomStack
      screenOptions={
        {
          // Refer to the React Navigation docs https://reactnavigation.org/docs/stack-navigator
        }
      }
    />
  );
}
```

### Tabs

Компонент навигации, который предоставляет навигационное меню

Структура:

app/

- \_layout.js
- (tabs)/
- - \_layout.js
- - (home)/
- - - index.js
- - - details.js
- - - \_layout.js

```js
// app/_layout.js
import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen name="(tabs)" />
    </Stack>
  );
}


// app/(tabs)/_layout.js
import { Tabs } from "expo-router";

export default function TabLayout() {
  return (
    <Tabs>
      <Tabs.Screen name="(home)" />
      <Tabs.Screen name="settings" />
    </Tabs>
  );
}
```

```js
// app/_Layout.tsx
// верхний layout над табами
import { Stack } from "expo-router/stack";

export default function AppLayout() {
  return (
    <Stack>
      {/* !!ожидает группу в папке tabs если в options не указать headerShown:
      false то появится второй заголовок, если это обернутый роут */}
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
    </Stack>
  );
}
```

```js
// layout для настройки табов
//app/(tabs)/_layout.tsx
import React from "react";
import FontAwesome from "@expo/vector-icons/FontAwesome";
import { Tabs } from "expo-router";

export default function TabLayout() {
  return (
    <Tabs screenOptions={{ tabBarActiveTintColor: "blue" }}>
      <Tabs.Screen name="tab1" options={{}} />
      <Tabs.Screen name="tab2" options={{}} />
      // динамический роут для таб
      <Tabs.Screen
        //имя динамического роута
        name="[user]"
        options={{
          href: {
            pathname: "/[user]",
            params: {
              user: "evanbacon",
            },
          },
        }}
      />
    </Tabs>
  );
}
```

<!--Навигация. хуки--------------------------------------------------------------------------------------------------------------------->

## Навигация. Хуки

### useFonts

```js
//устанавливаем: expo-font expo install expo-font
import { useFonts } from "expo-font";
...

//
    const [fontsLoaded] = useFonts({
    "open-sans": require("./assets/fonts/OpenSans-Regular.ttf"),
    "open-sans-bold": require("./assets/fonts/OpenSans-Bold.ttf"),
  });

  if (!fontsLoaded) return <AppLoading />;

  //использование в компонентах

  const styles = StyleSheet.create({
  title: {
    fontFamily: "open-sans-bold",
    ....
  },
});
```

### useFocusEffect

вызовет функцию если роут активный

```js

import { useFocusEffect } from 'expo-router';
import { useCallback } from 'react';

export default function Route() {
  useFocusEffect(
    useCallback(() => {
      console.log('Hello')
    }, []);
  );

  return </>;
}
```

### useGlobalSearchParams

возвращает глобальный url вне зависимости от компонента, обновляется на каждое изменения

```js
// файл app/profile/[user].tsx
// acme://profile/userName?extra=info

import { Text } from "react-native";

import { useGlobalSearchParams } from "expo-router";

export default function Route() {
  const { user, extra } = useGlobalSearchParams<{ user: string; query?: string }>();

  // user === userName, extra === info

  return <Text>User: {user}</Text>;
}
```

### useLocalSearchParams

возвращает параметры поиска для компонента, они обновятся когда глобальный url соответствует роуту

```js
// useLocalSearchParams хук позволяет работать с search params
import { useLocalSearchParams } from "expo-router";

export default function Page() {
  const { slug } = useLocalSearchParams<{ user: string; query?: string }>();

  return <Text>Blog post: {slug}</Text>;
}
```

### useNavigation

useNavigation хук, который возвращает объект с методами:

- setOptions - устанавливает настройки роута
- navigate - осуществляет навигацию

```js
import { Stack, useNavigation } from "expo-router";
import { Text, View } from "react-native";

export default function Home() {
  const navigation = useNavigation();

  React.useEffect(() => {
    navigation.setOptions({ headerShown: false });
  }, [navigation]);

  return (
    <View _style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
      <Text>Home Screen</Text>
    </View>
  );
}
```

```js
//можно передать адрес в аргумента в качестве layout
const parentLayout = useNavigation("/orders/menu");
```

### usePathname

```js
import { usePathname } from "expo-router";

export default function Route() {
  const pathname = usePathname();

  return <Text>User: {user}</Text>;
}
```

### useRouter

```js
import { View, Text } from "react-native";
import { Stack, useLocalSearchParams, useRouter } from "expo-router";

export default function Details() {
  const router = useRouter();

  // проверка на возможность очистки стека роутов
  if (router.canDismiss()) {
    // очистит стек от последнего роута
    router.dismiss(count); //count -любое положительное число
    // очистит весь стек
    router.dismissAll();
  }

  const replace = () => router.replace("/profile/settings");

  return (
    <View>
      <Stack.Screen options={{ title: params.name }} />
      <Text onPress={() => router.setParams({ name: "Updated" })}>
        Update the title
      </Text>
    </View>
  );
}
```

### useSegments

позволяет определить сегмент

```js
import { Text } from "react-native";
import { useSegments } from "expo-router";

export default function Route() {
  const segments = useSegments<['profile'] | ['profile', '[user]']>();

  return <Text>Hello</Text>;
}
```

<!--Контексты--------------------------------------------------------------------------------------------------------------------->

## Навигация. Контексты

### Тема

```js
import {
  ThemeProvider,
  DarkTheme,
  DefaultTheme,
  useTheme,
} from "@react-navigation/native";
import { Slot } from "expo-router";

export default function RootLayout() {
  return (
    //контекст темы
    <ThemeProvider value={DarkTheme}>
      <Slot />
    </ThemeProvider>
  );
}
```

<!--Вспомогательные функции--------------------------------------------------------------------------------------------------------------------->

## Навигация. Вспомогательные функции

### router

навигацию можно осуществлять с помощью route Объекта. Методы объекта: navigate, push, replace, back, canGoBack, setParams. Поддерживает автокомплит

```js
import { router } from "expo-router";

export function logout() {
  router.replace("/login"); // можно осуществлять навигацию push или navigate
  router.setParams({ name: "" }); // менять настройки текущего роута
  router.back();// Вернуть назад
  router.canGoBack();
  router.dismiss(count: Number);// удалить из стека роуты
  router.dismissAll();// удалить все из стека роуты
  router.canDismiss();//можно ли удалить роуты
}
```

### SplashScreen

```js
//app/_layout.js
import { SplashScreen, Slot } from "expo-router";
import { useFonts, Inter_500Medium } from "@expo-google-fonts/inter";

// предотвращаем закрытие SplashScreen
SplashScreen.preventAutoHideAsync();

export default function Layout() {
  const [fontsLoaded, fontError] = useFonts({
    Inter_500Medium,
  });

  useEffect(() => {
    if (fontsLoaded || fontError) {
      // если все загрузилось, то закрываем SplashScreen
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded, fontError]);

  if (!fontsLoaded && !fontError) {
    return null;
  }

  return <Slot />;
}
```

```js
// общий паттерн
import { Text } from "react-native";
import * as SplashScreen from "expo-splash-screen";

SplashScreen.preventAutoHideAsync();

export default function App() {
  const [isReady, setReady] = React.useState(false);

  React.useEffect(() => {
    setTimeout(() => {
      SplashScreen.hideAsync();
      setReady(true);
    }, 1000);
  }, []);

  return <Text>My App</Text>;
}
```

### API Routes

позволяет добавить апи методы

```js
// api json
{
  "web": {
    "bundler": "metro",
    "output": "server"
  }
  "plugins": [
    [
      "expo-router",
      {
        "origin": "https://evanbacon.dev/"
      }
    ]
  ]

}
//app hello+api.ts
export function GET(request: Request) {
  return Response.json({ hello: 'world' });
}
// app/blog/[post]+api,ts
export async function GET(request: Request, { post }: Record<string, string>) {
  // const postId = request.expoUrl.searchParams.get('post')
  // fetch data for 'post'
if (!post) {
    return new Response('No post found', {
      status: 404,
      headers: {
        'Content-Type': 'text/plain',
      },
    });
  }

  const body = await request.json();

  return Response.json({ ... });
}
// app.index
import { Button } from 'react-native';
// ф-ция для доступа
async function fetchHello() {
  const response = await fetch('/hello');
  const data = await response.json();
  alert('Hello ' + data.hello);
}

export default function App() {
  return <Button onPress={() => fetchHello()} title="Fetch hello" />;
}

```

### Sitemap

откроет карту

```js
// запустить следующую команду
// npx uri-scheme open exp://192.168.87.39:19000/--/form-sheet --ios
// app/_sitemap.tsx
export default function Sitemap() {
  return null;
}
```

<!--утилиты--------------------------------------------------------------------------------------------------------------------->

# Утилиты

## expo-image-picker

позволяет работать с изображениями на устройстве

```js
import {
  launchCameraAsync,
  useCameraPermissions,
  PermissionStatus,
} from "expo-image-picker";

// хук дял работы с изображениями возвращает два параметра
// cameraPermissionInformation - информация о доступе к устройству
// requestPermission - функция для запроса доступа
const [cameraPermissionInformation, requestPermission] = useCameraPermissions();
// функция для статуса доступа
const verifyPermissions = async () => {
  // PermissionStatus - enum со значениями статуса
  if (cameraPermissionInformation.status === PermissionStatus.UNDETERMINED) {
    const permissionResponse = await requestPermission();

    return permissionResponse.granted;
  }

  if (cameraPermissionInformation.status === PermissionStatus.DENIED) {
    Alert.alert("camera is blocked", "You need permission");
    return false;
  }

  return true;
};
// функция для работы с изображением
const takeImageHandler = async () => {
  // запрос доступа
  const hasPermission = await verifyPermissions();
  if (!hasPermission) {
    return;
  }

  // работа с изображением
  const image = await launchCameraAsync({
    allowsEditing: true,
    aspect: [16, 9],
    quality: 0.5,
  });

  setPickedImage(image.assets[0].uri);
  onTakeImage(image.uri);
};
```

## expo-location

```js
import {
  getCurrentPositionAsync,
  useForegroundPermissions,
  PermissionStatus,
} from "expo-location";

const LocationPicker = ({ onPickLocation }) => {
  //зук для работы с доступом
  const [locationPermissionInformation, requestPermission] =
    useForegroundPermissions();

  const verifyPermissions = async () => {
    if (
      locationPermissionInformation.status === PermissionStatus.UNDETERMINED
    ) {
      const permissionResponse = await requestPermission();
      return permissionResponse.granted;
    }
    if (locationPermissionInformation.status === PermissionStatus.DENIED) {
      Alert.alert("map is blocked", "You need permission");
      return false;
    }
    return true;
  };

  // функция для работы с геоданными
  const getLocationHandler = async () => {
    const hasPermission = await verifyPermissions();

    if (!hasPermission) return;

    const location = await getCurrentPositionAsync();

    setPickedLocation({
      lat: location.coords.latitude,
      long: location.coords.longitude,
    });
  };

  const pickOnMapHandler = () => {
    navigation.navigate("Map");
  };

  let locationPreview = <Text>No location picked yet</Text>;

  if (pickedLocation) {
    locationPreview = (
      <Image
        source={require("../../assets/map.png")}
        style={styles.mapPreviewImage}
      />
    );
  }
};
```

<!--Компоненты--------------------------------------------------------------------------------------------------------------------->

# Компоненты

## KeyboardAvoidingView

- компонент уводит экран вверх, при вводе с клавиатуры, нужно обернуть всю страницу

```js
<KeyboardAvoidingView
  behavior={"height" | "position" | "padding"} //поведение при появлении клавиатуры
  contentContainerStyle={type ViewStyle} //стиль контейнера при behavior === "position"
  enabled={Boolean}
  keyboardVerticalOffset={Number}
/>
```

## SafeAreaProvider

Предоставляет для iOS и Android оболочку. которая позволяет обернуть приложение для избежания попадания элементов Ui на камеры, которые встроены в экран

```js
// npx expo install react-native-safe-area-context
import { View, Text } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";

// оборачиваем в провайдер
export default function App() {
  return (
    <SafeAreaProvider>
      <HomeScreen />
    </SafeAreaProvider>
  );
}

function HomeScreen() {
  // используем padding
  const insets = useSafeAreaInsets();
  return (
    <View _style={{ flex: 1, paddingTop: insets.top }}>
      <Text _style={{ fontSize: 28 }}>Content is in safe area.</Text>
    </View>
  );
}
```

## SplashScreen

SplashScreen настраивается в app.json

```ts
{
  "splash": {
    "image": "./assets/images/splash.png", // ссылка на картинку
    "backgroundColor": "#FEF9B0", // цвет фона
    "resizeMode": "cover" | 'contain',
  }
}
```

## StatusBar

позволяет изменить верхнее меню телефона

```js
<StatusBar _style={"auto" | "inverted" | "light" | "dark"} />
```

<!--Хуки--------------------------------------------------------------------------------------------------------------------->

```js
import { Appearance, useColorScheme } from "react-native";

function MyComponent() {
  let colorScheme = useColorScheme();

  if (colorScheme === "dark") {
    // render some dark thing
  } else {
    // render some light thing
  }
}
```

<!--Работа со статикой--------------------------------------------------------------------------------------------------------------------->

# Работа со статикой

## Fonts

- использование с помощью app.json
- с помощью хука

```js
// Rest of the import statements
import { useFonts } from "expo-font";

export default function App() {
  const [fontsLoaded] = useFonts({
    "Inter-Black": require("./assets/fonts/Inter-Black.otf"),
  });

  <Text styles={{ fontFamily: "Inter-Black", fontSize: 30 }}>Inter Black</Text>;
}
```

- использование google fonts

```js
// npx expo install expo-font @expo-google-fonts/inter

//app.json

{
  "plugins": [
    [
      "expo-font",
      {
        "fonts": ["node_modules/@expo-google-fonts/inter/Inter_100Thin.ttf"]
      }
    ]
  ]
}

```

- загрузка из web

```js
import { useFonts } from "expo-font";

export default function App() {
  const [fontsLoaded] = useFonts({
    "Inter-SemiBoldItalic":
      "https://rsms.me/inter/font-files/Inter-SemiBoldItalic.otf?v=3.12",
  });

  if (!fontsLoaded) {
    return null;
  }

  return (
    <View>
      <Text styles={{ fontFamily: "Inter-SemiBoldItalic", fontSize: 30 }}>
        Inter SemiBoldItalic
      </Text>
      <Text>Platform Default</Text>
    </View>
  );
}
```

## Assets

- можно загружать в runtime

```js
import { useAssets } from "expo-asset";

export default function App() {
  const [assets, error] = useAssets([
    require("path/to/example-1.jpg"),
    require("path/to/example-2.png"),
  ]);

  return assets ? <Image source={assets[0]} /> : null;
  // загрузка через Ui
  return (
    <Image
      source={{ uri: "https://example.com/logo.png" }}
      style={{ width: 50, height: 50 }}
    />
  );
}
```

<!--хуки--------------------------------------------------------------------------------------------------------------------->

# Хуки

## useAnimatedStyle

Анимации осуществляются с помощью библиотеки react-native-reanimated

```js
//npx expo install react-native-reanimated
import Animated, {
  useSharedValue,
  withTiming,
  useAnimatedStyle,
  Easing,
} from "react-native-reanimated";
import { View, Button, StyleSheet } from "react-native";

export default function AnimatedStyleUpdateExample() {
  const randomWidth = useSharedValue(10);

  const config = {
    duration: 500,
    easing: Easing.bezier(0.5, 0.01, 0, 1),
  };

  const style = useAnimatedStyle(() => {
    return {
      width: withTiming(randomWidth.value, config),
    };
  });

  return (
    <View>
      <Animated.View style={[styles.box, style]} />
      <Button
        title="toggle"
        onPress={() => {
          randomWidth.value = Math.random() * 350;
        }}
      />
    </View>
  );
}
```

## useColorScheme

возвращает цвет темы

```js
import { Text, StyleSheet, View, useColorScheme } from "react-native";
import { StatusBar } from "expo-status-bar"; // Automatically switches bar style based on theme.

export default function App() {
  const colorScheme = useColorScheme();

  const themeTextStyle =
    colorScheme === "light" ? styles.lightThemeText : styles.darkThemeText;
  const themeContainerStyle =
    colorScheme === "light" ? styles.lightContainer : styles.darkContainer;

  return (
    <View>
      <Text>Color scheme: {colorScheme}</Text>
      <StatusBar />
    </View>
  );
}
```
