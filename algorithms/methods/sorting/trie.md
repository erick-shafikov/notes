# Бор (префиксное дерево, Trie)

Бор — структура данных для строк, а не сортировка. Путь от корня соответствует префиксу; пометка в узле означает конец полного слова. Общий префикс хранится один раз. Для слов `кот`, `код` общим будет путь `к → о`.

```ts
class TrieNode {
  children = new Map<string, TrieNode>();
  terminal = false;
}

class Trie {
  private root = new TrieNode();

  insert(word: string): void {
    let node = this.root;
    for (const char of word) { // Обход Unicode-кодовых точек.
      let child = node.children.get(char);
      if (!child) {
        child = new TrieNode();
        node.children.set(char, child);
      }
      node = child;
    }
    node.terminal = true;
  }

  private find(prefix: string): TrieNode | undefined {
    let node = this.root;
    for (const char of prefix) {
      const child = node.children.get(char);
      if (!child) return undefined;
      node = child;
    }
    return node;
  }

  has(word: string): boolean {
    return this.find(word)?.terminal ?? false;
  }

  startsWith(prefix: string): boolean {
    return this.find(prefix) !== undefined;
  }
}

const trie = new Trie();
trie.insert("кот");
trie.insert("код");
console.log(trie.has("ко")); // false: префикс не является полным словом.
console.log(trie.startsWith("ко")); // true
console.log(trie.has("кот")); // true
trie.insert("");
console.log(trie.has("")); // true: терминальным стал корень.
```

При ожидаемом `O(1)` доступе к `Map` вставка, поиск слова и поиск префикса требуют `O(L)`, где `L` — длина строки в кодовых точках. Память `O(S)`, где `S` — суммарная длина добавленных слов; общие префиксы уменьшают число узлов. Повторная вставка не создаёт копию слова: эта версия хранит множество, а не частоты.

Бор используют для автодополнения, словарей и как основу [Ахо — Корасика](aho-corasick.md). Обход терминальных узлов с упорядоченными рёбрами даёт лексикографический порядок слов. Сам `Map` хранит порядок вставки, поэтому без сортировки рёбер такой порядок не гарантирован. Нормализация Unicode и регистр здесь не изменяются: `е` и `ё`, а также разные формы записи символа различаются.
