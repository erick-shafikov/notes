# Platform

**свойства**

- OS: 'ios' | 'android' | 'native' | 'default'
- Version

```js
import { Platform, StyleSheet } from "react-native";

const styles = StyleSheet.create({
  height: Platform.OS === "ios" ? 200 : 100,
  //альтернативный вариант через Platform.select borderWidth: Platform.OS === "android" ? 2 : 0,
  borderWidth: Platform.select({ iso: 0, android: 2 }),
});
```

```js
//разные компоненты для разных платформ, select позволяет вернуть то значение, которую нужно будет для ios или android
const Component = Platform.select({
  ios: () => require("ComponentIOS"),
  android: () => require("ComponentAndroid"),
})();
<Component />;
```
