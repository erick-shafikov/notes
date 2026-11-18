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
        view-model/<slug>-view-model.ts
        hooks/use-<slug>-view-model.ts
      index.ts
  pages/
    <entity>-page/
      hooks/use-<entity>-payment.ts
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
  index.ts
```

Когда добавлять остальное:

| Файл | Когда появляется |
| --- | --- |
| `model/<slug>-types.ts` | доменный тип отличается от типа, сгенерированного из OpenAPI |
| `api/<slug>-dto.ts` | формат ответа отличается от доменного типа (snake_case, даты строками, вложенность) или ответ нужно валидировать в рантайме |
| `model/<slug>-form-schema.ts` | у сущности появилась форма |
| `model/<slug>-rules.ts` | появилось первое правило или инвариант, нужные больше чем в одном месте |
| `ui/constants`, `ui/errors` | у статусов появились лейблы и цвета, у ошибок — тексты |
| `ui/view-model` + `ui/hooks` | производной логики экрана больше, чем чтение полей (`data.title`) |
| `pages/<entity>-page/hooks` | на странице появился сценарий из нескольких фич |

Пока view-model нет, производные значения считаются в `select` у `useQuery`, а хук фичи просто возвращает результат запроса:

```ts
import { useQuery } from "@tanstack/react-query";
import { EntityStatus } from "../../model/<slug>-constants";
import { entityQueryOptions } from "../../api/<slug>-queries";

export const useEntity = (id: string) =>
  useQuery({
    ...entityQueryOptions(id),
    select: (entity) => ({ ...entity, isPaid: entity.status === EntityStatus.PAID }),
  });
```

Когда `select` разрастается или в нём появляются флаги, лейблы и ошибки, его содержимое переезжает в `buildEntityViewModel`.

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
import type { EntityStatus } from "./<slug>-constants";

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
import { EntityStatus } from "./<slug>-constants";
import type { Entity } from "./<slug>-types";

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
import { EntityStatus } from "../model/<slug>-constants";
import type { Entity } from "../model/<slug>-types";

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
import { entityDtoSchema, toEntity } from "./<slug>-dto";
import type { EntityFormValues } from "../model/<slug>-form-schema";

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
import { entityGateway } from "./<slug>-gateway";

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
import { EntityStatus } from "../../model/<slug>-constants";

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

#### `view-model/<slug>-view-model.ts`

View-model — готовое состояние экрана для компонентов. Это чистая функция: на входе данные запроса, статус и флаги, на выходе значения и флаги для UI. Заменяет класс `EntityRepresentation`: класс наследовал запросы и API, поэтому логика знала про TanStack и `fetch`. Функцию можно тестировать без React и без моков сети.

```ts
import { formatAmount } from "@/shared/helpers/number-helpers";
import { formatDate } from "@/shared/helpers/string-helpers";
import { ApiError } from "@/infrastructure/api/api-client";
import { isEntityEditable, isEntityPayable } from "../../model/<slug>-rules";
import type { Entity } from "../../model/<slug>-types";
import { ENTITY_STATUS_VIEW } from "../constants/<slug>-ui-constants";
import { getEntityErrorMessage } from "../errors/<slug>-error-messages";

type BuildEntityViewModelParams = {
  entity: Entity | undefined;
  isPending: boolean;
  error: Error | null;
  canEditFlag: boolean;
};

export const buildEntityViewModel = ({
  entity,
  isPending,
  error,
  canEditFlag,
}: BuildEntityViewModelParams) => ({
  isLoading: isPending,
  errorMessage: error
    ? getEntityErrorMessage(error instanceof ApiError ? error.code : undefined)
    : null,
  title: entity?.title ?? "",
  amountText: entity ? formatAmount(entity.amount) : "",
  createdAtText: entity ? formatDate(entity.createdAt) : "",
  statusView: entity ? ENTITY_STATUS_VIEW[entity.status] : null,
  isPayable: entity ? isEntityPayable(entity) : false,
  isEditable: canEditFlag && entity ? isEntityEditable(entity) : false,
});

export type EntityViewModel = ReturnType<typeof buildEntityViewModel>;
```

#### `hooks/use-<slug>-view-model.ts`

Единственное место фичи, где встречаются TanStack Query и view-model: пара «запрос → логика».

```ts
import { useQuery } from "@tanstack/react-query";
import { entityQueryOptions } from "../../api/<slug>-queries";
import { buildEntityViewModel } from "../view-model/<slug>-view-model";

export const useEntityViewModel = (id: string, canEditFlag: boolean) => {
  const { data, isPending, error } = useQuery(entityQueryOptions(id));

  return buildEntityViewModel({ entity: data, isPending, error, canEditFlag });
};
```

Переиспользуемый UI-сценарий фичи (компонент + свой хук, например модалка редактирования) тоже лежит в `ui/`, рядом с этими хуками.

### index.ts

Публичный API фичи. Снаружи импортируют только отсюда, глубокие импорты (`@/features/entity/api/...`) запрещены линтером (`no-restricted-imports` или `eslint-plugin-boundaries`).

```ts
export { EntityStatus } from "./model/<slug>-constants";
export type { Entity } from "./model/<slug>-types";
export { entityKeys, entityQueryOptions } from "./api/<slug>-queries";
export { useEntityViewModel } from "./ui/hooks/use-<slug>-view-model";
export type { EntityViewModel } from "./ui/view-model/<slug>-view-model";
```

# pages

Страница — composition root. Здесь соединяются параметры роута, feature-флаги, view-model разных фич и React-состояние.

## pages/\<entity\>-page

### hooks/use-\<entity\>-payment.ts

Логика другой фичи в том же UI-пространстве: на странице сущности есть оплата этой сущности. Фичи не наследуют и не импортируют друг друга. Сценарий, который их связывает, собирается на странице композицией двух view-model.

```ts
import { useEntityViewModel } from "@/features/entity";
import { usePaymentViewModel } from "@/features/payment";

export const useEntityPayment = (id: string, canEditFlag: boolean) => {
  const entity = useEntityViewModel(id, canEditFlag);
  const payment = usePaymentViewModel(id);

  return {
    entity,
    payment,
    canPay: entity.isPayable && payment.isAvailable,
  };
};
```

Если такой сценарий понадобится на нескольких страницах, он переезжает в отдельную фичу (например, `features/entity-payment`), которая зависит от `entity` и `payment` через их `index.ts`.

### \<entity\>-page.tsx

- `EntityPage` читает роут и флаги и проверяет `id`: `useParams` возвращает `string | undefined`.
- `EntityPageContent` получает готовые значения. UI-состояние (открыта ли модалка) остаётся в React. Производные значения считаются при рендере, а не копируются в `useState` через `useEffect`.

```tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useEntityFeatureFlags } from "@/infrastructure/feature-flags/hooks/use-entity-feature-flags";
import { useEntityPayment } from "./hooks/use-entity-payment";

type EntityPageContentProps = {
  id: string;
  canEditFlag: boolean;
};

const EntityPageContent = ({ id, canEditFlag }: EntityPageContentProps) => {
  const [isPayModalOpen, setPayModalOpen] = useState(false);
  const { entity, payment, canPay } = useEntityPayment(id, canEditFlag);

  if (entity.isLoading) {
    return <PageSkeleton />;
  }

  if (entity.errorMessage) {
    return <PageError message={entity.errorMessage} />;
  }

  return (
    <PageLayout title={entity.title}>
      <EntityStatusBadge view={entity.statusView} />
      <EntityDetails
        amount={entity.amountText}
        createdAt={entity.createdAtText}
      />
      <PayButton disabled={!canPay} onClick={() => setPayModalOpen(true)} />
      <PayModal
        open={isPayModalOpen}
        payment={payment}
        onClose={() => setPayModalOpen(false)}
      />
    </PageLayout>
  );
};

export const EntityPage = () => {
  const { id } = useParams<{ id: string }>();
  const { canEdit } = useEntityFeatureFlags();

  if (!id) {
    return <NotFound />;
  }

  return <EntityPageContent id={id} canEditFlag={canEdit} />;
};
```

Где лежит логика сценариев:

- логика одной страницы — в `pages/<entity>-page/hooks`;
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
import { ApiCode } from "./api-codes";

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
import { handlers } from "./handlers";

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
