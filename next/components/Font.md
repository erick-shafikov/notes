# Font

```tsx
import { Roboto } from "next/font/google"; //импорт конкретного шрифта
const roboto = Roboto({ weight: ["300", "500"], subsets: ["latin"] }); //настройка

const Layout = () => <body className={roboto.className}></body>; //подключение
```

```tsx
// Возможно подключение нескольких шрифтов
// Подключение локальных шрифтов:
import localFont from "next/font/local";

const myFont = localFont({
  src: "./my-font.woff2",
  weight: ["100", "200"], //или '100 900' от 100 до 900
  display: "swap",
  variable: "--font-inter", //объявление в качестве переменной
  style: "normal", // 'italic','oblique'
  subsets: ["latin"],
  axes: ["slnt"],
  display: "swap", //"auto", 'block', 'swap', 'fallback', 'optional'
  preload: true,
  fallback: ["system-ui", "arial"],
  adjustFontFallback: false; //'Times New Roman
  variable: "--my-font",
  // @font-face
  declarations: [{ prop: 'ascent-override', value: '90%' }]
});

export const App = () => <html lang="en" className={myFont.className}></html>;
//или
export const App = () => <html lang="en" style={inter.style}></html>;
```
