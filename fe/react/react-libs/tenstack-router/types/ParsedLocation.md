```ts
interface ParsedLocation {
  href: string;
  pathname: string;
  search: TFullSearchSchema;
  searchStr: string;
  state: ParsedHistoryState;
  hash: string;
  maskedLocation?: ParsedLocation;
  unmaskOnReload?: boolean;
}
```
