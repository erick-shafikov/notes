# TS

```ts
export class LinkedListNode<T> {
  value: T;
  next: LinkedListNode<T> | null;
  prev: LinkedListNode<T> | null;

  constructor(value: T) {
    this.value = value;
    this.next = null;
    this.prev = null;
  }
}

export class DoublyLinkedList<T> {
  head: LinkedListNode<T> | null;
  tail: LinkedListNode<T> | null;
  current: LinkedListNode<T> | null;

  constructor() {
    this.head = null;
    this.tail = null;
    this.current = null;
  }

  add(value: T): void {
    const newNode = new LinkedListNode(value);
    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
      this.current = newNode;
    } else {
      if (this.tail) {
        this.tail.next = newNode;
        newNode.prev = this.tail;
        this.tail = newNode;
      }
    }
  }

  removeCurrent(): void {
    if (!this.current) return;

    if (this.current.prev) {
      this.current.prev.next = this.current.next;
    } else {
      this.head = this.current.next;
    }

    if (this.current.next) {
      this.current.next.prev = this.current.prev;
    } else {
      this.tail = this.current.prev;
    }

    this.current = this.current.next || this.current.prev || null;
  }

  next(): void {
    if (this.current && this.current.next) {
      this.current = this.current.next;
    } else {
      this.current = this.head;
    }
  }

  prev(): void {
    if (this.current && this.current.prev) {
      this.current = this.current.prev;
    } else {
      this.current = this.tail;
    }
  }
}
```

# двусвязный список

```ts
class Node<T> {
  data: T;
  next: Node<T> | null = null;
  prev: Node<T> | null = null;

  constructor(data: T) {
    this.data = data;
  }
}

export class LinkedList<T> {
  private head: Node<T> | null = null;
  private current: Node<T> | null = null;

  add(data: T): void {
    const newNode = new Node(data);

    if (!this.head) {
      this.head = newNode;
      this.current = this.head;
    } else if (this.current) {
      this.current.next = newNode;
      newNode.prev = this.current;
      this.current = newNode;
    }
  }

  undo(): T | null {
    if (this.current && this.current.prev) {
      this.current = this.current.prev;
      return this.current.data;
    }
    return null;
  }

  redo(): T | null {
    if (this.current && this.current.next) {
      this.current = this.current.next;
      return this.current.data;
    }
    return null;
  }

  canUndo(): boolean {
    return !!this.current && !!this.current.prev;
  }

  canRedo(): boolean {
    return !!this.current && !!this.current.next;
  }

  getCurrentData(): T | null {
    return this.current ? this.current.data : null;
  }
}
```
