# ValidateLinkOptions

типизация пропсов с помощью ValidateLinkOptions

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
