# linkOptions

так как если создать dashboardLinkOptions с помощью объекта, то придется подгонять поля под типизацию

```tsx
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
