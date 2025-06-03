/* СТРАТЕГИЯ 
    Шаблон «Стратегия» позволяет переключаться между алгоритмами или стратегиями в зависимости от ситуации.
    Шаблон «Стратегия» позволяет при выполнении выбирать поведение алгоритма.
*/

interface SortStrategy {
    sort(dataset: any[]): any[];
}

class BubbleSortStrategy implements SortStrategy {
    public sort(dataset: any[]): any[] {
        console.log('Sorting using bubble sort');

        // Do sorting
        return dataset;
    }
}

class QuickSortStrategy implements SortStrategy {
    public sort(dataset: any[]): any[] {
        console.log('Sorting using quick sort');

        // Do sorting
        return dataset;
    }
}

class Sorter {
    protected sorter: BubbleSortStrategy | QuickSortStrategy;

    constructor(sorter: SortStrategy) {
        this.sorter = sorter;
    }

    public sort(dataset: any[]): any[] {
        return this.sorter.sort(dataset);
    }
}

const dataset = [1, 5, 4, 3, 2, 8];

const sorter1 = new Sorter(new BubbleSortStrategy());
sorter1.sort(dataset); // Output : Пузырьковая сортировка

const sorter2 = new Sorter(new QuickSortStrategy());
sorter2.sort(dataset); // Output : Быстрая сортировка
