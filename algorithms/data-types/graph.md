```ts
const graph = {
  a: {
    siblings: ["b"],
  },

  b: {
    siblings: ["a", "c"],
  },

  c: {
    siblings: ["a"],
  },
};
```

# ts

```ts
class Graph {
  nodes: { id: string; x?: number; y?: number; fx?: number; fy?: number }[] =
    [];
  links: { source: string; target: string }[] = [];

  addNode(id: string) {
    this.nodes.push({ id });
    this.update();
  }

  addLink(source: string, target: string) {
    this.links.push({ source, target });
    this.update();
  }
}
```
