# ИНТЕРПРЕТАТОР

Шаблон «Интерпретатор» определяет грамматику простого языка и предоставляет интерпретатор для обработки предложений на этом языке. Каждое правило грамматики представлено классом. Применяется, например, в регулярных выражениях, SQL-парсерах и вычислителях выражений.

Interpreter (Интерпретатор) - Реализует грамматику языка, представляя каждое правило отдельным классом.
`Vocabulary.Add(expressionA); Vocabulary.Add(expressionB); Vocabulary.Translate();`

```ts
// Контекст — хранит переменные и их значения
class Context {
  private variables: Map<string, number> = new Map();

  set(name: string, value: number) {
    this.variables.set(name, value);
  }

  get(name: string): number {
    if (!this.variables.has(name)) {
      throw new Error(`Переменная '${name}' не определена`);
    }
    return this.variables.get(name)!;
  }
}

// Абстрактное выражение
interface Expression {
  interpret(context: Context): number;
}

// Терминальное выражение — число
class NumberExpression implements Expression {
  constructor(private value: number) {}

  interpret(_context: Context): number {
    return this.value;
  }
}

// Терминальное выражение — переменная
class VariableExpression implements Expression {
  constructor(private name: string) {}

  interpret(context: Context): number {
    return context.get(this.name);
  }
}

// Нетерминальное выражение — сложение
class AddExpression implements Expression {
  constructor(
    private left: Expression,
    private right: Expression,
  ) {}

  interpret(context: Context): number {
    return this.left.interpret(context) + this.right.interpret(context);
  }
}

// Нетерминальное выражение — вычитание
class SubtractExpression implements Expression {
  constructor(
    private left: Expression,
    private right: Expression,
  ) {}

  interpret(context: Context): number {
    return this.left.interpret(context) - this.right.interpret(context);
  }
}

// Нетерминальное выражение — умножение
class MultiplyExpression implements Expression {
  constructor(
    private left: Expression,
    private right: Expression,
  ) {}

  interpret(context: Context): number {
    return this.left.interpret(context) * this.right.interpret(context);
  }
}

// Использование
const context = new Context();
context.set("x", 10);
context.set("y", 5);

// Строим AST для выражения: (x + y) * (x - 3)
const expression = new MultiplyExpression(
  new AddExpression(new VariableExpression("x"), new VariableExpression("y")),
  new SubtractExpression(new VariableExpression("x"), new NumberExpression(3)),
);

console.log(expression.interpret(context)); // (10 + 5) * (10 - 3) = 15 * 7 = 105
```
