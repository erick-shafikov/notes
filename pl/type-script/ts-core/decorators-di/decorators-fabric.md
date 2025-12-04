# Фабрика декораторов

Вся суть в замыкании и возврате функции

```ts
function retry2(maxRetryAttempts: number, sleepTime: number) {
  return function (target: any, context: any) {
    const resultMethod = async function (this: any, ...args: any[]) {
      let lastError = undefined;

      for (let attemptNum = 1; attemptNum <= maxRetryAttempts; attemptNum++) {
        try {
          return await target.apply(this, args);
        } catch (error) {
          lastError = error;

          if (attemptNum < maxRetryAttempts) {
            await sleep(sleepTime);
          }
        }
      }

      throw lastError;
    };

    return resultMethod;
  };
}

// применение
@retry2(3, 500)
```
