# RouteEvent

```ts
type RouterEvents = {
  onBeforeNavigate: {
    type: "onBeforeNavigate";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
    pathChanged: boolean;
    hrefChanged: boolean;
  };
  onBeforeLoad: {
    type: "onBeforeLoad";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
    pathChanged: boolean;
    hrefChanged: boolean;
  };
  onLoad: {
    type: "onLoad";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
    pathChanged: boolean;
    hrefChanged: boolean;
  };
  onResolved: {
    type: "onResolved";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
    pathChanged: boolean;
    hrefChanged: boolean;
  };
  onBeforeRouteMount: {
    type: "onBeforeRouteMount";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
    pathChanged: boolean;
    hrefChanged: boolean;
  };
  onInjectedHtml: {
    type: "onInjectedHtml";
    promise: Promise<string>;
  };
  onRendered: {
    type: "onRendered";
    fromLocation?: ParsedLocation;
    toLocation: ParsedLocation;
  };
};
```
