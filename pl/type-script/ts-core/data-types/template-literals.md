# template literal types

```ts
type ReadOrWrite = "read" | "write";
type Bulk = "bulk" | "write"; //пример
type Access1 = `can${ReadOrWrite}`;
/* type Access1 = "canRead" | "canWrite" */
type Access2 = `can${Capitalize<ReadOrWrite>}`;
/* type Access2 = "canRead" | "canWrite" */
type Access3 = `can${Capitalize<ReadOrWrite>}${Capitalize<Bulk>}`;
/* type Access3 = "canReadWrite" | "canReadBulk" | "canWriteWrite" | "canWriteBulk" */
type ErrorOrSuccess = "error" | "success";
type ResponseT = {
  result: `http${Capitalize<ErrorOrSuccess>}`;
};
const a: ResponseT = {
  result: "httpError",
};
type ReadOrWriteBulk<T> = T extends `can${infer R}` ? R : never; //infer - вытащить
type T = ReadOrWriteBulk<Access3>;
//type T = "ReadWrite" | "ReadBulk" | "WriteWrite" | "WriteBulk"
```
