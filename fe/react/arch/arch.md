# Архитектура React-приложения

- **Логика** — всё, что касается сущностей: типы, статусы, правила, вычисления. Без низкоуровневого API и API фреймворка.
- **Отображение** — всё, что работает с API браузера и UI-фреймворка: хуки, состояние, компоненты, запросы через TanStack Query.
- **Сценарий** — цепочка связанных событий (открытие модалки → действие → отображение результата).

Логика и отображение соединяются только на уровне страницы. Зависимости идут в одну сторону:

```text
model  ←  api  ←  ui (view-model, hooks)  ←  pages
```

- `model` не импортирует React, TanStack Query и `fetch`;
- `api` знает про транспорт и TanStack Query, но не про React-компоненты;
- `ui` фичи знает про свою фичу, но не про другие фичи;
- `pages` — composition root: собирает фичи, флаги, параметры роута и связывает сценарии разных фич.

## Обзор структуры

```text
app/
  features/
    <entity>/
      model/
        <slug>-constants.ts
        <slug>-types.ts
        <slug>-form-schema.ts
        <slug>-rules.ts
      api/
        <slug>-dto.ts
        <slug>-gateway.ts
        <slug>-queries.ts
      ui/
        constants/<slug>-ui-constants.ts
        errors/<slug>-error-messages.ts
        view-model/<entity>-view-model.ts
        view-model/<slug>-view-model.ts
        hooks/use-<slug>-query.ts
      index.ts
  pages/
    <entity>-page/
      hooks/use-<entity>-payment.ts
      <entity>-route.tsx
      <entity>-page.tsx
  shared/
    helpers/
      string-helpers.ts
      number-helpers.ts
      crypto-helpers.ts
    libs/
      validation/
        string-validation-schemas.ts
        number-validation-schemas.ts
      query/
        query-client.ts
  infrastructure/
    api/
      api-client.ts
      api-types.ts
      api-codes.ts
    mocks/
      handlers.ts
      browser.ts
    sentry/
      sentry-client.ts
    feature-flags/
      hooks/use-entity-feature-flags.ts
      constants.ts
```

## Файлы по необходимости

Дерево выше показывает полную фичу. Новая фича начинается с минимума, остальные файлы появляются, когда для них есть содержимое. Так структура не превращается в boilerplate.

Минимальная фича:

```text
features/<entity>/
  model/<slug>-constants.ts
  api/<slug>-gateway.ts
  api/<slug>-queries.ts
  ui/view-model/<slug>-view-model.ts
  ui/hooks/use-<slug>-query.ts
  index.ts
```

Всё, что страница берёт из сущности, идёт через методы view-model среза: даже если сейчас это один метод, у страницы сразу есть одно место, куда добавляются новые.

Когда добавлять остальное:

| Файл | Когда появляется |
| --- | --- |
| `model/<slug>-types.ts` | доменный тип отличается от типа, сгенерированного из OpenAPI |
| `api/<slug>-dto.ts` | формат ответа отличается от доменного типа (snake_case, даты строками, вложенность) или ответ нужно валидировать в рантайме |
| `model/<slug>-form-schema.ts` | у сущности появилась форма |
| `model/<slug>-rules.ts` | появилось первое правило или инвариант, нужные больше чем в одном месте |
| `ui/constants`, `ui/errors` | у статусов появились лейблы и цвета, у ошибок — тексты |
| `ui/view-model/<entity>-view-model.ts` | у сущности появился второй срез, и ему нужны те же методы, что и первому |
| `pages/<entity>-page/hooks` | на странице появился сценарий из нескольких фич |

## Импорты

Все импорты — через алиас `@/`, относительные пути (`./`, `../../`) не используются: файл можно перенести, не переписывая импорты, а по пути сразу видно, из какого слоя он импортирует.

```ts
import { EntityStatus } from "@/features/<entity>/model/<slug>-constants";
import { entityQueryOptions } from "@/features/<entity>/api/<slug>-queries";
```

Импорт с полным путём до файла допустим только внутри своей фичи. Снаружи фичу импортируют через её `index.ts` (`@/features/<entity>`).

# features

Одна папка на сущность или бизнес-область (`order`, `payment`, …). Внутри `<slug>` — конкретный срез сущности (например, `order-details`, `order-list`).

## features/\<entity\>

### model

Чистая логика сущности. Раньше здесь был `repository`, но repository обычно означает доступ к данным, поэтому папка переименована в `model`. Тексты ошибок отсюда ушли в `ui/errors`, транспортные коды — в `api`.

#### `<slug>-constants.ts`

API-константы: статусы, состояния. Без названий и цветов — они в `ui/constants`.

```ts
export const EntityStatus = {
  DRAFT: "draft",
  ACTIVE: "active",
  PAID: "paid",
} as const;

export type EntityStatus = (typeof EntityStatus)[keyof typeof EntityStatus];
```

#### `<slug>-types.ts`

Доменный тип: то, с чем работает приложение, а не сырой ответ API.

```ts
import type { EntityStatus } from "@/features/<entity>/model/<slug>-constants";

export type Entity = {
  id: string;
  title: string;
  amount: number;
  status: EntityStatus;
  createdAt: Date;
};
```

#### `<slug>-form-schema.ts`

Схема формы (данные, которые пользователь отправляет в запросе). Собирается из общих схем `shared/libs/validation`.

```ts
import { z } from "zod";
import { requiredString } from "@/shared/libs/validation/string-validation-schemas";
import { positiveAmount } from "@/shared/libs/validation/number-validation-schemas";

export const entityFormSchema = z.object({
  title: requiredString,
  amount: positiveAmount,
});

export type EntityFormValues = z.infer<typeof entityFormSchema>;
```

#### `<slug>-rules.ts`

Инварианты и вычисления над сущностью — чистые функции, тестируются без моков.

```ts
import { EntityStatus } from "@/features/<entity>/model/<slug>-constants";
import type { Entity } from "@/features/<entity>/model/<slug>-types";

export const isEntityPayable = (entity: Entity) =>
  entity.status === EntityStatus.ACTIVE && entity.amount > 0;

export const isEntityEditable = (entity: Entity) =>
  entity.status === EntityStatus.DRAFT;
```

### api

Транспорт и кэш. Раньше здесь были `services` с классами `EntitySlugApiService` и агрегатором `EntityApiService extends A, B`. В TS нет множественного наследования, а агрегатор не нужен: публичный набор функций отдаёт `index.ts`.

#### `<slug>-dto.ts`

Схема ответа API (отдельная от схемы формы) и маппер DTO → доменный тип. Сырой ответ не уходит дальше этого файла.

```ts
import { z } from "zod";
import { EntityStatus } from "@/features/<entity>/model/<slug>-constants";
import type { Entity } from "@/features/<entity>/model/<slug>-types";

export const entityDtoSchema = z.object({
  id: z.string(),
  title: z.string(),
  amount: z.number(),
  status: z.nativeEnum(EntityStatus),
  created_at: z.string(),
});

export type EntityDto = z.infer<typeof entityDtoSchema>;

export const toEntity = (dto: EntityDto): Entity => ({
  id: dto.id,
  title: dto.title,
  amount: dto.amount,
  status: dto.status,
  createdAt: new Date(dto.created_at),
});
```

#### `<slug>-gateway.ts`

Запросы поверх общего `apiClient`. Ошибки не глотаются — их пробрасывает `apiClient` как `ApiError`. `signal` передаётся для отмены запроса.

```ts
import { apiClient } from "@/infrastructure/api/api-client";
import { entityDtoSchema, toEntity } from "@/features/<entity>/api/<slug>-dto";
import type { EntityFormValues } from "@/features/<entity>/model/<slug>-form-schema";

export const entityGateway = {
  async getById(id: string, signal?: AbortSignal) {
    const dto = await apiClient.get(`/entities/${id}`, { signal });
    return toEntity(entityDtoSchema.parse(dto));
  },

  async update(id: string, values: EntityFormValues) {
    const dto = await apiClient.put(`/entities/${id}`, { body: values });
    return toEntity(entityDtoSchema.parse(dto));
  },
};
```

#### `<slug>-queries.ts`

Фабрика ключей и `queryOptions()` из TanStack Query v5. Заменяет класс `EntitySlugQuery` и декоратор-адаптер: `queryOptions` уже типизирует пару `queryKey` + `queryFn`. `queryFn` — функция, а не результат вызова.

```ts
import { queryOptions } from "@tanstack/react-query";
import { entityGateway } from "@/features/<entity>/api/<slug>-gateway";

export const entityKeys = {
  all: ["entity"] as const,
  detail: (id: string) => [...entityKeys.all, "detail", id] as const,
};

export const entityQueryOptions = (id: string) =>
  queryOptions({
    queryKey: entityKeys.detail(id),
    queryFn: ({ signal }) => entityGateway.getById(id, signal),
  });
```

Инвалидация после мутации — через те же ключи: `queryClient.invalidateQueries({ queryKey: entityKeys.detail(id) })`.

### ui

Отображение фичи: всё, что нужно React-компонентам, но относится только к этой сущности.

#### `constants/<slug>-ui-constants.ts`

Динамичные лейблы, цвета, названия табов.

```ts
import { EntityStatus } from "@/features/<entity>/model/<slug>-constants";

export const ENTITY_STATUS_VIEW: Record<
  EntityStatus,
  { label: string; color: string }
> = {
  [EntityStatus.DRAFT]: { label: "Черновик", color: "gray" },
  [EntityStatus.ACTIVE]: { label: "Активна", color: "blue" },
  [EntityStatus.PAID]: { label: "Оплачена", color: "green" },
};

export const ENTITY_TABS = ["details", "history"] as const;
```

#### `errors/<slug>-error-messages.ts`

Тексты ошибок по коду. Коды приходят из `ApiError.code` и `infrastructure/api/api-codes.ts`; тексты относятся к отображению (и к i18n, если он есть).

```ts
import { ApiCode } from "@/infrastructure/api/api-codes";

export const ENTITY_ERROR_MESSAGES: Partial<Record<ApiCode, string>> = {
  [ApiCode.NOT_FOUND]: "Сущность не найдена",
  [ApiCode.FORBIDDEN]: "Нет доступа к сущности",
};

export const getEntityErrorMessage = (code: ApiCode | undefined) =>
  (code && ENTITY_ERROR_MESSAGES[code]) ?? "Что-то пошло не так";
```

#### view-model

View-model — обычный класс, который собирает в одном месте все методы, нужные странице для работы с сущностью: форматирование, статусы, тексты ошибок, проверки прав и правил. Компоненты не разбирают сущность сами и не импортируют по отдельности хелперы, правила и UI-константы — они вызывают методы view-model.

Что внутри класса:

- методы, которые принимают данные (сущность, ошибку) аргументами и возвращают готовые значения для UI;
- константы фичи, привязанные к статическим полям (`static readonly STATUS_VIEW = ENTITY_STATUS_VIEW`), чтобы страница брала их из того же объекта;
- конструктор — только если все методы завязаны на одни и те же параметры экрана, например `id` и feature-флаг. Данные запроса в конструктор не передаются.

От старого класса `EntityRepresentation` остаётся идея «всё для страницы в одном объекте», но уходит наследование от запросов и API: из-за него логика знала про TanStack и `fetch`. Класс можно тестировать через `new` без React и без моков сети.

View-model делится на два уровня:

| Класс | Файл | Что внутри | Где используется |
| --- | --- | --- | --- |
| `EntityViewModel` | `view-model/<entity>-view-model.ts` | методы и константы, переиспользуемые всеми срезами сущности | только внутри фичи, как родитель view-model срезов |
| `EntitySlugViewModel` | `view-model/<slug>-view-model.ts` | наследует `EntityViewModel` и добавляет методы, нужные только этому срезу | экземпляр создаёт роут страницы и передаёт вниз ссылкой |

```text
EntityViewModel          общие методы и константы сущности
   ↑ extends
EntitySlugViewModel      методы конкретной страницы
   ↓ new (в роуте)
pages/<entity>-page      viewModel пропсом → хуки страницы и фичи
```

В TS нет множественного наследования, поэтому у view-model среза один родитель — `EntityViewModel`. Всё остальное (`shared/helpers`, правила из `model`, UI-константы, тексты ошибок) не наследуется, а импортируется как переиспользуемые утилиты.

Пока срез один, `EntityViewModel` не нужен: все методы живут в `EntitySlugViewModel`. Когда появляется второй срез и ему нужны те же методы, они переезжают в `EntityViewModel`.

##### `view-model/<entity>-view-model.ts`

Переиспользуемые методы и константы сущности для всех срезов. Конструктора нет: класс не знает ни про конкретную страницу, ни про запрос.

```ts
import { formatAmount } from "@/shared/helpers/number-helpers";
import { formatDate } from "@/shared/helpers/string-helpers";
import { ApiError } from "@/infrastructure/api/api-client";
import { isEntityPayable } from "@/features/<entity>/model/<slug>-rules";
import type { Entity } from "@/features/<entity>/model/<slug>-types";
import { ENTITY_STATUS_VIEW, ENTITY_TABS } from "@/features/<entity>/ui/constants/<slug>-ui-constants";
import { getEntityErrorMessage } from "@/features/<entity>/ui/errors/<slug>-error-messages";

export class EntityViewModel {
  static readonly STATUS_VIEW = ENTITY_STATUS_VIEW;
  static readonly TABS = ENTITY_TABS;

  getAmountText(entity: Entity) {
    return formatAmount(entity.amount);
  }

  getCreatedAtText(entity: Entity) {
    return formatDate(entity.createdAt);
  }

  getStatusView(entity: Entity) {
    return EntityViewModel.STATUS_VIEW[entity.status];
  }

  getErrorMessage(error: Error) {
    return getEntityErrorMessage(error instanceof ApiError ? error.code : undefined);
  }

  isPayable(entity: Entity) {
    return isEntityPayable(entity);
  }
}
```

##### `view-model/<slug>-view-model.ts`

View-model конкретного среза (например, `order-details`), с которым работает страница. Наследует `EntityViewModel` и добавляет методы этой страницы. Здесь все методы завязаны на `id` и флаг редактирования, поэтому они приходят в конструктор.

```ts
import { isEntityEditable } from "@/features/<entity>/model/<slug>-rules";
import type { Entity } from "@/features/<entity>/model/<slug>-types";
import { EntityViewModel } from "@/features/<entity>/ui/view-model/<entity>-view-model";

export class EntitySlugViewModel extends EntityViewModel {
  constructor(
    readonly id: string,
    private readonly canEditFlag: boolean,
  ) {
    super();
  }

  getPageTitle(entity: Entity) {
    return `${entity.title} — ${this.getStatusView(entity).label}`;
  }

  getEditUrl() {
    return `/entities/${this.id}/edit`;
  }

  canEdit(entity: Entity) {
    return this.canEditFlag && isEntityEditable(entity);
  }

  canPay(entity: Entity, isPaymentAvailable: boolean) {
    return this.isPayable(entity) && isPaymentAvailable;
  }
}
```

Другой срез (например, `order-list`) так же наследует `EntityViewModel` и добавляет свои методы: пагинацию, выделение, пустое состояние.

#### `hooks/use-<slug>-query.ts`

Хуки фичи не создают view-model, а получают готовый экземпляр аргументом и берут из него то, что нужно для запроса (`id`).

```ts
import { useQuery } from "@tanstack/react-query";
import { entityQueryOptions } from "@/features/<entity>/api/<slug>-queries";
import type { EntitySlugViewModel } from "@/features/<entity>/ui/view-model/<slug>-view-model";

export const useEntitySlugQuery = (viewModel: EntitySlugViewModel) =>
  useQuery(entityQueryOptions(viewModel.id));
```

Переиспользуемый UI-сценарий фичи (компонент + свой хук, например модалка редактирования) тоже лежит в `ui/`, рядом с этими хуками.

### index.ts

Публичный API фичи. Снаружи импортируют только отсюда (`@/features/<entity>`), глубокие импорты из чужой фичи (`@/features/<entity>/api/...`) запрещены линтером (`no-restricted-imports` или `eslint-plugin-boundaries`). `EntityViewModel` наружу не экспортируется: страницы работают с view-model среза. `EntitySlugViewModel` экспортируется как значение, потому что экземпляр создаёт роут.

```ts
export { EntityStatus } from "@/features/<entity>/model/<slug>-constants";
export type { Entity } from "@/features/<entity>/model/<slug>-types";
export { entityKeys, entityQueryOptions } from "@/features/<entity>/api/<slug>-queries";
export { useEntitySlugQuery } from "@/features/<entity>/ui/hooks/use-<slug>-query";
export { EntitySlugViewModel } from "@/features/<entity>/ui/view-model/<slug>-view-model";
```

# pages

Страница — composition root. Здесь соединяются параметры роута, feature-флаги, view-model разных фич и React-состояние.

## pages/\<entity\>-page

Экземпляр view-model создаётся один раз наверху — в роуте — и спускается вниз ссылкой:

```text
<entity>-route.tsx        useParams + флаги → new EntitySlugViewModel(id, canEdit)
   ↓ viewModel (единственный пропс)
<entity>-page.tsx         рендер, UI-состояние
   ↓ viewModel (аргумент)
hooks/use-<entity>-payment.ts, useEntitySlugQuery, ...
```

- страница получает view-model единственным пропсом;
- хуки страницы и фичи не создают и не импортируют экземпляры view-model, а получают тот же экземпляр аргументом;
- компоненты ниже получают либо готовые значения, либо тот же `viewModel`.

### Где хранить экземпляр: `useMemo`, а не `useRef`

View-model — набор чистых функций плюс неизменяемые `id` и флаг. Состояния у него нет, поэтому вопрос только в том, когда создавать новый экземпляр:

- `useRef` для этого не подходит. Ref нужен для изменяемых значений, которые не участвуют в рендере, а view-model участвует в каждом рендере. Ref создаётся один раз и не узнаёт о смене `id` или флага: при переходе с `/entities/1` на `/entities/2` React переиспользует компонент роута, и в ref останется старый `id`.
- `useMemo` по `[id, canEdit]` создаёт новый экземпляр ровно тогда, когда меняются параметры конструктора, и сохраняет ссылку между остальными рендерами. Стабильная ссылка важна для `memo`-компонентов и массивов зависимостей хуков ниже.
- Если у класса нет параметров конструктора, React-хуки не нужны вовсе: экземпляр создаётся один раз на уровне модуля (`export const entityViewModel = new EntityViewModel()`).

Создать экземпляр дорого не стоит, поэтому `useMemo` здесь нужен не ради производительности, а ради стабильной ссылки.

### \<entity\>-route.tsx

Единственное место, где читаются роут и флаги и создаётся view-model. Хуки вызываются до проверки `id`, потому что ранний `return` перед хуками нарушает правила хуков.

```tsx
import { useMemo } from "react";
import { useParams } from "react-router-dom";
import { EntitySlugViewModel } from "@/features/entity";
import { useEntityFeatureFlags } from "@/infrastructure/feature-flags/hooks/use-entity-feature-flags";
import { EntityPage } from "@/pages/<entity>-page/<entity>-page";

export const EntityRoute = () => {
  const { id } = useParams<{ id: string }>();
  const { canEdit } = useEntityFeatureFlags();

  const viewModel = useMemo(
    () => (id ? new EntitySlugViewModel(id, canEdit) : null),
    [id, canEdit],
  );

  if (!viewModel) {
    return <NotFound />;
  }

  return <EntityPage viewModel={viewModel} />;
};
```

### hooks/use-\<entity\>-payment.ts

Логика другой фичи в том же UI-пространстве: на странице сущности есть оплата этой сущности. Фичи не наследуют и не импортируют друг друга. Сценарий, который их связывает, собирается на странице. Хук получает view-model аргументом от страницы.

```ts
import { useEntitySlugQuery, type EntitySlugViewModel } from "@/features/entity";
import { usePaymentQuery } from "@/features/payment";

export const useEntityPayment = (viewModel: EntitySlugViewModel) => {
  const query = useEntitySlugQuery(viewModel);
  const payment = usePaymentQuery(viewModel.id);

  return {
    query,
    payment,
    canPay:
      query.data && payment.data
        ? viewModel.canPay(query.data, payment.data.isAvailable)
        : false,
  };
};
```

Если такой сценарий понадобится на нескольких страницах, он переезжает в отдельную фичу (например, `features/entity-payment`), которая зависит от `entity` и `payment` через их `index.ts`.

### \<entity\>-page.tsx

Страница получает view-model единственным пропсом, передаёт его в свои хуки и передаёт данные запроса в методы view-model. UI-состояние (открыта ли модалка) остаётся в React. Производные значения считаются методами view-model при рендере, а не копируются в `useState` через `useEffect`.

```tsx
import { useState } from "react";
import type { EntitySlugViewModel } from "@/features/entity";
import { useEntityPayment } from "@/pages/<entity>-page/hooks/use-<entity>-payment";

type EntityPageProps = {
  viewModel: EntitySlugViewModel;
};

export const EntityPage = ({ viewModel }: EntityPageProps) => {
  const [isPayModalOpen, setPayModalOpen] = useState(false);
  const { query, payment, canPay } = useEntityPayment(viewModel);

  if (query.isPending) {
    return <PageSkeleton />;
  }

  if (query.isError) {
    return <PageError message={viewModel.getErrorMessage(query.error)} />;
  }

  const entity = query.data;

  return (
    <PageLayout title={viewModel.getPageTitle(entity)}>
      <EntityStatusBadge view={viewModel.getStatusView(entity)} />
      <EntityDetails
        amount={viewModel.getAmountText(entity)}
        createdAt={viewModel.getCreatedAtText(entity)}
      />
      {viewModel.canEdit(entity) && <EditButton href={viewModel.getEditUrl()} />}
      <PayButton disabled={!canPay} onClick={() => setPayModalOpen(true)} />
      <PayModal
        open={isPayModalOpen}
        payment={payment.data}
        onClose={() => setPayModalOpen(false)}
      />
    </PageLayout>
  );
};
```

Где лежит логика сценариев:

- создание view-model — в `pages/<entity>-page/<entity>-route.tsx`, один раз на страницу;
- логика одной страницы — в `pages/<entity>-page/hooks`, хуки получают view-model аргументом;
- методы сущности, общие для нескольких срезов, — в `EntityViewModel` (`features/<entity>/ui/view-model/<entity>-view-model.ts`);
- методы конкретной страницы — в `EntitySlugViewModel` (`features/<entity>/ui/view-model/<slug>-view-model.ts`), который наследует `EntityViewModel`;
- переиспользуемая UI-логика одной фичи (компонент + хук) — в `features/<entity>/ui`;
- сценарий нескольких фич — на странице, при переиспользовании — в отдельной фиче-связке.

# shared

Код, который не знает ни о каких сущностях. Сущностные запросы, ключи и константы сюда не попадают.

## helpers

Чистые функции вместо классов-миксинов (`StringHelpers`, `DataHelpers`): их импортируют там, где нужно, а не наследуют.

### `string-helpers.ts`

```ts
export const formatDate = (date: Date) =>
  new Intl.DateTimeFormat("ru-RU").format(date);

export const truncate = (value: string, max: number) =>
  value.length > max ? `${value.slice(0, max)}…` : value;
```

### `number-helpers.ts`

```ts
export const formatAmount = (value: number, currency = "RUB") =>
  new Intl.NumberFormat("ru-RU", { style: "currency", currency }).format(value);
```

### `crypto-helpers.ts`

## libs

### validation

Общие кирпичики схем, из которых фичи собирают свои схемы форм.

#### `string-validation-schemas.ts`

```ts
import { z } from "zod";

export const requiredString = z.string().trim().min(1);
```

#### `number-validation-schemas.ts`

```ts
import { z } from "zod";

export const positiveAmount = z.number().positive();
```

### query

#### `query-client.ts`

Единственный `QueryClient` приложения с общими настройками.

```ts
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
});
```

# infrastructure

Интеграции с внешним миром: HTTP, мониторинг, флаги, моки.

## api

### `api-client.ts`

Общий клиент: базовый URL из env, проверка `response.ok`, типизированная ошибка, поддержка `signal`. Переключение API (окружение, провайдер) происходит здесь, в одном месте, а не в каждой фиче.

```ts
import { ApiCode } from "@/infrastructure/api/api-codes";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: ApiCode | undefined,
  ) {
    super(`API error ${status}`);
  }
}

type RequestOptions = {
  body?: unknown;
  signal?: AbortSignal;
};

const BASE_URL = import.meta.env.VITE_API_URL;

const request = async (
  method: string,
  path: string,
  options: RequestOptions = {},
) => {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, payload?.code);
  }

  return response.status === 204 ? null : ((await response.json()) as unknown);
};

export const apiClient = {
  get: (path: string, options?: RequestOptions) =>
    request("GET", path, options),
  post: (path: string, options?: RequestOptions) =>
    request("POST", path, options),
  put: (path: string, options?: RequestOptions) =>
    request("PUT", path, options),
  delete: (path: string, options?: RequestOptions) =>
    request("DELETE", path, options),
};
```

### `api-types.ts`

Общие типы API: пагинация, обёртки ответов.

### `api-codes.ts`

Общие коды ошибок API.

```ts
export const ApiCode = {
  NOT_FOUND: "not_found",
  FORBIDDEN: "forbidden",
  VALIDATION: "validation_error",
} as const;

export type ApiCode = (typeof ApiCode)[keyof typeof ApiCode];
```

## mocks

Фейковое API для разработки, тестов и Storybook — MSW вместо `fake-api-service` в каждой фиче. Код фич не меняется: перехват идёт на уровне сети.

### `handlers.ts`

```ts
import { http, HttpResponse } from "msw";

export const handlers = [
  http.get("*/entities/:id", ({ params }) =>
    HttpResponse.json({
      id: params.id,
      title: "Тестовая сущность",
      amount: 1000,
      status: "active",
      created_at: "2026-01-01T00:00:00Z",
    }),
  ),
];
```

### `browser.ts`

Запускается в точке входа только при `import.meta.env.VITE_API_MOCKS === "true"`.

```ts
import { setupWorker } from "msw/browser";
import { handlers } from "@/infrastructure/mocks/handlers";

export const worker = setupWorker(...handlers);
```

## sentry

### `sentry-client.ts`

## feature-flags

### `hooks/use-entity-feature-flags.ts`

### `constants.ts`

# todo

- SSR: если появится, решить, где граница server/client и где префетчить `entityQueryOptions`.
- Настроить линтер на направление зависимостей (`model ← api ← ui ← pages`) и запрет глубоких импортов.
