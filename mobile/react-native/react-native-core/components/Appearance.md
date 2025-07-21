# Appearance

Позволяет узнать настройки телефона

```js
import { Appearance } from "react-native";

const colorScheme = Appearance.getColorScheme();
if (colorScheme === "dark") {
  // Use dark color scheme
}

//методы

getColorScheme(); // вернет 'light' | 'dark' | null;
setColorScheme("light" | "dark" | null);
addChangeListener(
  listener: (preferences: {colorScheme: 'light' | 'dark' | null}) => void,
): NativeEventSubscription;
```
