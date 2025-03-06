# Font

```tsx
import { Roboto } from "next/font/google"; //импорт конкретного шрифта
const roboto = Roboto({ weight: ["300", "500"], subsets: ["latin"] }); //настройка

const Layout = () => <body className={roboto.className}></body>; //подключение
```

```tsx
// Возможно подключение нескольких шрифтов
// Подключение локальных шрифтов:
import localFont from 'next/font/local'

const myFont = localFont({
  src: './my-font.woff2',
  display: 'swap',
  variable: '--font-inter', //объявление в качестве переменной
})

<html lang="en" className={myFont.className}> </html>

```
