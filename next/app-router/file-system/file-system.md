# Типы файлов

## папки

pages - страницы
api - папка для api методов
\_file - папка внутри app,

## Файлы

- default.js – компонент, который

- page.js – страница
- layout.js – компонент обертка
- error.js – компонент для отображении ошибки, должен быть CC, эмитирует или может быть заменен ErrorBoundary
- global-error.js – компонент глобальной ошибки
- loading.js – компонент загрузки компонент который имитирует Suspense загрузку, можно эмитировать обернув в Suspense подгружаемые компоненты Обертка над множеством страниц. Сохраняет состояние. Должен быть RootLayout с тегами html и body. По умолчанию серверный компонент, но может быть и клиентским. Невозможна передача данных от Layout к дочерним компонентам (можно решить через fetch одинаковых данных).
- not-found.js – компонент not found
- route.ts – путь api
- RootLayout – обязательный компонент, должен иметь html и body, могут быть и вложенные
- template.js – обертка, которая создается каждый раз новая

Комбинации файлов

- route groups – (groupe) – папка, позволяет объединить в себе несколько страниц в общую папку
- параллельные роуты – отображение осуществляется в layout в слотах props.path1 props.path2, где path1 и path2 – это файлы @path1 @path2. Могут выступать как conditional routes
- Перехватывающие роуты – когда при переходе на страницу появляется другая страница, а по прямым ссылкам попадаем на те pages, на которые запланировали попасть
- \_lib – приватные роуты

- middleware

Можно создавать папки с компонентами, хелперами вне папки app. Три стратегии организации проекта:

- все внутри app
- все вне app
- Хранить все внутри роутов (feature-sliced)

Роутинг можно осуществлять window.history.pushState

<!-- document.js --------------------------------------------------------------------------------------------------------------------------->

## \_document.js

- \_document.jd - может обновлять html и body теги страницы

- !!!строго серверный компонент
- cte такой же как и в getInitialProps

renderPage- необходима для библиотек css-in-js

```tsx
import Document, {
  Html,
  Head,
  Main,
  NextScript,
  DocumentContext,
  DocumentInitialProps,
} from "next/document";

class MyDocument extends Document {
  static async getInitialProps(
    ctx: DocumentContext
  ): Promise<DocumentInitialProps> {
    const originalRenderPage = ctx.renderPage;

    // Run the React rendering logic synchronously
    ctx.renderPage = () =>
      originalRenderPage({
        // Useful for wrapping the whole react tree
        enhanceApp: (App) => App,
        // Useful for wrapping in a per-page basis
        enhanceComponent: (Component) => Component,
      });

    // Run the parent `getInitialProps`, it now includes the custom `renderPage`
    const initialProps = await Document.getInitialProps(ctx);

    return initialProps;
  }

  render() {
    return (
      <Html lang="en">
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
```
