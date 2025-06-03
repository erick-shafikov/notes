/* ХРАНИТЕЛЬ
Шаблон «Хранитель» фиксирует и хранит текущее состояние объекта, чтобы оно легко восстанавливалось.
Шаблон «Хранитель» позволяет восстанавливать объект в его предыдущем состоянии (отмена через откат — undo via rollback). 
*/

class EditorMemento {
    protected content: string;

    public constructor(content: string) {
        this.content = content;
    }

    public getContent() {
        return this.content;
    }
}

class Editor {
    protected content = '';

    public type(words: string) {
        this.content = this.content + ' ' + words;
    }

    public getContent() {
        return this.content;
    }

    public save() {
        return new EditorMemento(this.content);
    }

    public restore(memento: EditorMemento) {
        this.content = memento.getContent();
    }
}

const editor = new Editor();

// Пишем что-нибудь
editor.type('This is the first sentence.');
editor.type('This is second.');

// Сохранение состояния в: This is the first sentence. This is second.
const saved = editor.save();

// Пишем ещё
editor.type('And this is third.');

// Output: Содержимое до сохранения
console.log(editor.getContent()); // This is the first sentence. This is second. And this is third.

// Восстанавливаем последнее сохранённое состояние
editor.restore(saved);

editor.getContent(); // This is the first sentence. This is second.

export {};
