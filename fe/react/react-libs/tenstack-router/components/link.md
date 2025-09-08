# linkOptions

```tsx
// так как если создать dashboardLinkOptions с помощью объекта, то придется подгонять поля под типизацию
const dashboardLinkOptions = linkOptions({
  to: "/dashboard",
  search: { search: "" },
});

// или
const props = {
  to: "/posts/",
} as const satisfies LinkProps;

// использования для Link
function DashboardComponent() {
  return <Link {...dashboardLinkOptions} />;
}

// использования для перенаправления
export const Route = createFileRoute("/dashboard")({
  component: DashboardComponent,
  validateSearch: (input) => ({ search: input.search }),
  beforeLoad: () => {
    // can used in redirect
    throw redirect(dashboardLinkOptions);
  },
});
```

```tsx
// поддерживается массив
const options = linkOptions([
  {
    to: "/dashboard",
    label: "Summary",
    activeOptions: { exact: true },
  },
  {
    to: "/dashboard/invoices",
    label: "Invoices",
  },
  {
    to: "/dashboard/users",
    label: "Users",
  },
]);

function DashboardComponent() {
  return (
    <>
      {options.map((option) => {
        return (
          <Link
            {...option}
            key={option.to}
            activeProps={{ className: `font-bold` }}
            className="p-2"
          >
            {option.label}
          </Link>
        );
      })}
    </>
  );
}
```

# createLink

позволяет создать пользовательскую ссылку

```tsx
import * as React from "react";
import { createLink, LinkComponent } from "@tanstack/react-router";

interface BasicLinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  // Add any additional props you want to pass to the anchor element
}

const BasicLinkComponent = React.forwardRef<HTMLAnchorElement, BasicLinkProps>(
  (props, ref) => {
    return (
      <a ref={ref} {...props} className={"block px-3 py-2 text-blue-700"} />
    );
  }
);

const CreatedLinkComponent = createLink(BasicLinkComponent);

export const CustomLink: LinkComponent<typeof BasicLinkComponent> = (props) => {
  return <CreatedLinkComponent preload={"intent"} {...props} />;
};
```

# типизация пропсов с помощью ValidateLinkOptions

Есть так же:

- ValidateLinkOptionsArray
- ValidateRedirectOptions
- ValidateNavigateOptions

```tsx
export interface HeaderLinkProps<
  TRouter extends RegisteredRouter = RegisteredRouter,
  TOptions = unknown
> {
  title: string;
  linkOptions: ValidateLinkOptions<TRouter, TOptions>;
}

export function HeadingLink<TRouter extends RegisteredRouter, TOptions>(
  props: HeaderLinkProps<TRouter, TOptions>
): React.ReactNode;
// перегрузка
export function HeadingLink(props: HeaderLinkProps): React.ReactNode {
  return (
    <>
      <h1>{props.title}</h1>
      <Link {...props.linkOptions} />
    </>
  );
}
```

# маскировка путей

```tsx
// скроет modal
const MaskLinkComponent = () => (
  <Link
    to="/photos/$photoId/modal"
    params={{ photoId: 5 }}
    mask={{
      to: "/photos/$photoId",
      params: {
        photoId: 5,
      },
    }}
  >
    Open Photo
  </Link>
);
// навигация
function onOpenPhoto() {
  navigate({
    to: "/photos/$photoId/modal",
    params: { photoId: 5 },
    mask: {
      to: "/photos/$photoId",
      params: {
        photoId: 5,
      },
    },
  });
}
```
