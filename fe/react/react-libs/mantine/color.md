# Цвета

каждый цвет находится в 10 состояниях, получить можно:

- из темы с помощью хука [useMantineTheme](./hooks/useMantineThemeHook.md)
- переменные css (--mantine-color-red-5)

[Добавляется в тему цвет](./objects/theme.md#colors)

Пропсы:

- пропс c - просто color
- пропс color - с более сложной логикой

# цветовые темы

- data-mantine-color-scheme атрибут на теге html
- lightHidden и darkHidden пропсы на компонентах позволяют скрывать компоненты при определенной теме

# функции

- darken - darken('rgb(245, 159, 0)', 0.5) - на 50% темнее
- lighten - lighten('#228BE6', 0.1) на 10% ярче
- alpha
- parseThemeColor - вернет объект вида

```ts
interface ParseThemeColorResult {
  isThemeColor: boolean;
  color: string;
  value: string;
  shade: MantineColorShade | undefined;
  variable: CssVariable | undefined;
}
```

- getThemeColor
- getGradient
- isLightColor
- luminance
