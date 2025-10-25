```ts
export interface ExpectStatic
  extends Chai.ExpectStatic,
    AsymmetricMatchersContaining {
  <T>(actual: T, message?: string): Assertion<T>;
  extend: (expects: MatchersObject) => void;
  anything: () => any;
  any: (constructor: unknown) => any;
  getState: () => MatcherState;
  setState: (state: Partial<MatcherState>) => void;
  not: AsymmetricMatchersContaining;
}
```
