# ActivityIndicator

вращающийся спиннер спиннер

```js
import { ActivityIndicator } from "react-native";

const App = () => (
  <ActivityIndicator
    animating={true} //крутится или нет
    color={"color"}
    hidesWhenStopped={true} //(IOS) отображать или нет пока крутится
    size={"large" | "small"} //размер
  />
);
```
