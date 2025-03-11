# getInitialProps (legacy)

асинхронная функция, отработает как на стороне сервера так и на стороне клиента, результат этой функции будет передан в компоненты. При начальной загрузке будет запущен только на сервере и при переходах

- !!! нельзя что бы данные были Date, Map, Set - только простые объекты
- !!! может быть использован только на верхнем уровне в \_app, не может быть использован во вложенных компонентах
- !!! так как выполняется на сервере и на клиенте не стоит передавать данные
- !!! если функция запускается в \_app то будет выполнена, даже если есть getServerSideProps при навигации

- [пример использования для передачи гидрированного состояния наверх](../../../../../notes-work/next-js/src/pages/_app.tsx.md)

возвращает объект ctx

```tsx
import { NextPageContext } from "next";

Page.getInitialProps = async (ctx: NextPageContext) => {
  const res = await fetch("https://api.github.com/repos/vercel/next.js");
  const json = await res.json();
  return { stars: json.stargazers_count };
};

export default function Page({ stars }: { stars: number }) {
  return stars;
}
```

```ts
type NextPageContext = {
  err?: Error;
  req?: IncomingMessage; //объект запроса
  res?: ServerResponse; // объект ответа
  pathname: string; //url запроса
  query: ParsedUrlQuery;
  asPath?: string; // фактического пути (включая запрос), отображаемого в браузере
};
```
