expect() — BDD-стиль (Behavior-Driven Development), для более простых тестов

```ts
import { assert, test } from "vitest";

test("add", () => {
  assert.equal(2 + 2, 4);
});
```

assert. — TDD-стиль (Test-Driven Development), для тестов с расширениям

```ts
import { expect, test } from "vitest";

test("add", () => {
  expect(2 + 2).toBe(4);
});
```

Методы assert.methods():

- assert.fail

```ts
import { assert, test } from "vitest";

test("assert.fail", () => {
  assert.fail("error message on failure");
  assert.fail("foo", "bar", "foo is not bar", "===");
});
```

- assert.isOk
- assert.isNotOk
- assert.equal
- assert.notEqual
- assert.strictEqual
- assert.deepEqual
- assert.notDeepEqual
- assert.isAbove
- assert.isAtLeast
- assert.isBelow
- assert.isAtMost
- assert.notEqual
- assert.isTrue
- assert.isNotTrue
- assert.isFalse
- assert.isNotFalse
- assert.isNull
- assert.isNotNull
- assert.isNaN
- assert.isNotNaN
- assert.exists
- assert.notExists
- assert.isUndefined
- assert.isDefined
- assert.isFunction
- assert.isNotFunction
- assert.isObject
- assert.isNotObject
- assert.isArray
- assert.isNotArray
- assert.isString
- assert.isNotString
- assert.isNumber
- assert.isNotNumber
- assert.isFinite
- assert.isBoolean
- assert.isNotBoolean
- assert.typeOf
- assert.notTypeOf
- assert.instanceOf
- assert.notInstanceOf
- assert.include
- assert.notInclude
- assert.deepInclude
- assert.notDeepInclude
- assert.nestedInclude
- assert.notNestedInclude
- assert.deepNestedInclude
- assert.notDeepNestedInclude
- assert.ownInclude
- assert.notOwnInclude
- assert.deepOwnInclude
- assert.notDeepOwnInclude
- assert.match
- assert.notMatch
- assert.property
- assert.notProperty
- assert.propertyVal
- assert.notPropertyVal
- assert.deepPropertyVal
- assert.notDeepPropertyVal
- assert.nestedProperty
- assert.notNestedProperty
- assert.nestedPropertyVal
- assert.notNestedPropertyVal
- assert.doesNotHaveAllKeys
- assert.hasAnyDeepKeys
- assert.hasAllDeepKeys
- assert.containsAllDeepKeys
- assert.doesNotHaveAnyDeepKeys
- assert.throws
- assert.doesNotThrow
- assert.operator
- assert.closeTo
- массивы:
- - assert.sameMembers
- - assert.notSameMembers
- - assert.sameDeepMembers
- - assert.notSameDeepMembers
- - assert.sameOrderedMembers
- - assert.notSameOrderedMembers
- - assert.sameDeepOrderedMembers
- - assert.notSameDeepOrderedMembers
- - assert.includeMembers
- - assert.notIncludeMembers
- - assert.includeDeepMembers
- - assert.notIncludeDeepMembers
- - assert.includeOrderedMembers
- - assert.notIncludeOrderedMembers
- - assert.includeDeepOrderedMembers
- - assert.notIncludeDeepOrderedMembers
- - assert.oneOf
- объекты и их свойства:
- - assert.changes
- - assert.changesBy
- - assert.doesNotChange
- - assert.changesButNotBy
- - assert.increases
- - assert.increasesBy
- - assert.doesNotIncrease
- - assert.increasesButNotBy
- - assert.decreases
- - assert.decreasesBy
- - assert.doesNotDecrease
- - assert.doesNotDecreaseBy
- - assert.decreasesButNotBy
- - assert.ifError
- - assert.isExtensible
- - assert.isNotExtensible
- - assert.isSealed
- - assert.isNotSealed
- - assert.isFrozen
- - assert.isNotFrozen
- - assert.isEmpty
- - assert.isNotEmpty
