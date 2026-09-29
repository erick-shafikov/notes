# Алгоритм Ахо — Корасик

Одновременно ищет много образцов в одном тексте. Это алгоритм поиска строк, а не сортировка. Основа — [бор](trie.md), дополненный суффиксными ссылками и ссылками на ближайший терминальный суффикс.

Суффиксная ссылка `fail` ведёт к самому длинному собственному суффиксу текущего пути, который также есть в боре. Если продолжить путь нельзя, переходим по `fail`, не возвращая указатель текста назад. Ссылка `output` позволяет сообщить совпадения с более короткими образцами.

```ts
type Match = { patternIndex: number; start: number; end: number };

class ACNode {
  next = new Map<string, number>();
  fail = 0;
  output = -1;
  ids: number[] = [];
}

function ahoCorasick(patterns: readonly string[], text: string): Match[] {
  const nodes: ACNode[] = [new ACNode()];
  const lengths: number[] = [];
  for (let id = 0; id < patterns.length; id++) {
    const chars = Array.from(patterns[id]);
    if (chars.length === 0) throw new RangeError("Пустые образцы не поддерживаются");
    lengths.push(chars.length);
    let state = 0;
    for (const char of chars) {
      let next = nodes[state].next.get(char);
      if (next === undefined) {
        next = nodes.length;
        nodes.push(new ACNode());
        nodes[state].next.set(char, next);
      }
      state = next;
    }
    nodes[state].ids.push(id); // Одинаковые образцы сохраняют отдельные ID.
  }

  const queue = [...nodes[0].next.values()];
  for (let head = 0; head < queue.length; head++) {
    const v = queue[head]; // BFS без дорогого Array.shift().
    for (const [char, child] of nodes[v].next) {
      let suffix = nodes[v].fail;
      while (suffix !== 0 && !nodes[suffix].next.has(char)) {
        suffix = nodes[suffix].fail;
      }
      const fail = nodes[suffix].next.get(char) ?? 0;
      nodes[child].fail = fail;
      nodes[child].output = nodes[fail].ids.length > 0 ? fail : nodes[fail].output;
      queue.push(child);
    }
  }

  const matches: Match[] = [];
  let state = 0;
  let position = 0;
  for (const char of text) {
    while (state !== 0 && !nodes[state].next.has(char)) state = nodes[state].fail;
    state = nodes[state].next.get(char) ?? 0;
    // Сначала текущее слово, затем его терминальные суффиксы.
    for (let v = state; v !== -1; v = nodes[v].output) {
      for (const id of nodes[v].ids) {
        matches.push({ patternIndex: id, start: position - lengths[id] + 1, end: position + 1 });
      }
    }
    position++;
  }
  return matches;
}

console.log(ahoCorasick(["he", "she", "his", "hers"], "ushers"));
// [
//   { patternIndex: 1, start: 1, end: 4 }, // she
//   { patternIndex: 0, start: 2, end: 4 }, // he — суффикс she
//   { patternIndex: 3, start: 2, end: 6 }, // hers
// ]
```

Индексы `start` и исключающая правая граница `end` измеряются в Unicode-кодовых точках. Для символов вне BMP они отличаются от UTF-16 индексов `String.slice`; выделять найденную строку можно через `Array.from(text).slice(start, end).join("")`. Перекрытия и повторяющиеся образцы поддерживаются; пустые образцы отклоняются явно.

Пусть `S` — суммарная длина образцов, `V` — число узлов, `L` — максимальная длина образца, `N` — длина текста, `Z` — число совпадений. При ожидаемом `O(1)` доступе к `Map` поиск занимает `O(N + Z)`: успешный переход увеличивает глубину на 1, а переход по `fail` уменьшает её. Память `O(S + Z)`, включая результат.

В этой компактной реализации построение имеет консервативную верхнюю оценку `O(S + VL)`: при вычислении каждой ссылки возможен проход по цепочке суффиксов. Полная таблица переходов для алфавита размера `σ` позволяет строить автомат за `O(S + Vσ)`, используя `O(Vσ)` памяти. Ссылки `output` вместо копирования всех списков суффиксных совпадений экономят память.

Применение: словарный поиск, фильтры ключевых слов, поиск множества сигнатур. Для одного образца часто достаточно более простого алгоритма. Связь с моделью автомата: [конечные автоматы](../../finite-automata.md).

Источник по реализации и представлению переходов: [Efficient implementation of the Aho–Corasick pattern matching automaton](https://www.cs.uku.fi/research/publications/reports/A-2005-2.pdf).
