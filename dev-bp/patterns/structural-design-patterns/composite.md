# компоновщик

Добавляет объекты в древовидные структуры для представления сложных иерархий. Позволяет работать с одиночными объектами и их группами одинаково. Он строит древовидную структуру:

- Leaf (лист) — простой объект
- Composite (контейнер) — содержит другие элементы (Leaf или Composite)

Клиент работает с ними через общий интерфейс.

```ts
interface FileSystemItem {
  getSize(): number;
}

// Лист — обычный файл
class File implements FileSystemItem {
  constructor(private size: number) {}

  public getSize(): number {
    return this.size;
  }
}

// Компоновщик — папка
class Directory implements FileSystemItem {
  private children: FileSystemItem[] = [];

  public add(item: FileSystemItem) {
    this.children.push(item);
  }

  public getSize(): number {
    return this.children.reduce((total, child) => {
      return total + child.getSize();
    }, 0);
  }
}

const file1 = new File(100);
const file2 = new File(200);

const folder1 = new Directory();
folder1.add(file1);
folder1.add(file2);

const file3 = new File(300);

const root = new Directory();
root.add(folder1);
root.add(file3);

// клиенту не важно — файл это или папка
console.log(root.getSize()); // 600
```

Клиент не различает файл это (File) или папка (Directory). React по сути работает как компоновщик - любой компонент может содержать другие компоненты, и рендер происходит одинаково
