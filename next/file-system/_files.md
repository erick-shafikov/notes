## Файлы

- page.js – страница
- layout.js – компонент обертка
- - RootLayout – обязательный компонент, должен иметь html и body, могут быть и вложенные
- error.js – компонент для отображении ошибки, должен быть CC, эмитирует или может быть заменен ErrorBoundary
- global-error.js – компонент глобальной ошибки
- loading.js – компонент загрузки компонент который имитирует Suspense загрузку, можно эмитировать обернув в Suspense подгружаемые компоненты Обертка над множеством страниц. Сохраняет состояние. Должен быть RootLayout с тегами html и body. По умолчанию серверный компонент, но может быть и клиентским. Невозможна передача данных от Layout к дочерним компонентам (можно решить через fetch одинаковых данных).
- not-found.js – компонент not found
- route.ts – путь api
- template.js – обертка, которая создается каждый раз новая

Директории

- route groups – (groupe) – папка, позволяет объединить в себе несколько страниц в общую папку
- параллельные роуты – отображение осуществляется в layout в слотах props.path1 props.path2, где path1 и path2 – это файлы @path1 @path2. Могут выступать как conditional routes
- Перехватывающие роуты – когда при переходе на страницу появляется другая страница, а по прямым ссылкам попадаем на те pages, на которые запланировали попасть
- \_lib – приватные роуты (даже если будут файлы page и route внутри)

- middleware

Можно создавать папки с компонентами, хелперами вне папки app. Три стратегии организации проекта:

- все внутри app
- все вне app
- Хранить все внутри роутов (feature-sliced)

Роутинг можно осуществлять window.history.pushState

# конфигурация сегмента

```tsx
export const experimental_ppr = undefined; //позволяет статические и динамические компоненты использовать вместе
// поменять на полностью динамические или полностью статично поведение
export const dynamic = "auto"; //'force-dynamic' - полностью динамический на каждый запрос, 'error', 'force-static - принудительно статическое';

export const dynamicParams = true; //если не сгенерированы в generateStaticParams, то сгенерируются при запросе при false - 404 error
//срок кеширования
export const revalidate = false; //0 - динамическая визуализация | number

export const fetchCache = "auto"; //"default-cache" |"only-cache" |"force-cache" |"force-no-store" |"default-no-store" |"only-no-store";
export const runtime = "nodejs"; // 'edge';платформа развертывания
export const preferredRegion = "auto"; // 'global' | 'home' | string | string[];
//ограничение выполнения логики на сервере
export const maxDuration = undefined; //number;
```
