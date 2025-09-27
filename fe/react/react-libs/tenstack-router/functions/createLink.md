# createLink

позволяет создать пользовательскую ссылку

```tsx
import * as React from "react";
import { createLink, LinkComponent } from "@tanstack/react-router";

interface BasicLinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  // дополнительные параметры ссылки
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
