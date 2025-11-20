```ts
class User {}
class Admin {}

type Entity = User | Admin;

function typeChecker(entity) {
  if(entity instance of User ){

  } else {
    // type entity === Admin
  }
}
```
