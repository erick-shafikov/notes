# useMantineColorScheme

```ts
type UseMantineColorScheme = ({ keepTransitions: boolean }) => {
  colorScheme: "dark" | "light" | "auto"; //текущая тема
  setColorScheme: (colorScheme: "dark" | "light" | "auto") => void; // сеттер
  toggleColorScheme: () => void; //переключатель
  clearColorScheme: () => void; // сброс до defaultColorScheme
};
```
