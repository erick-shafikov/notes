```js
const fetchFlights = (from) => {
  const map = { A: ["B", "C"], B: ["D", "E"], E: ["F"] };

  return map[from];
};

async function findPath(from, to, fetchFlights) {
  const queue = [from];
  const map = {};

  while (queue.length > 0) {
    const source = queue.pop();
    const targets = fetchFlights(source);

    if (!targets) {
      continue;
    }

    for (const target of targets) {
      queue.push(target);
      map[target] = source;

      if (target === to) {
        const result = [target];

        while (result[result.length - 1] !== from) {
          result.push(map[result[result.length - 1]]);
        }

        return result.reverse();
      }
    }
  }

  throw new Error("no way");
}
```
