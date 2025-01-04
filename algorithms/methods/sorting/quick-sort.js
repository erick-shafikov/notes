(function () {
  /**
   * Quick array sorting
   * @param {Array} array Source array
   * @returns {Array} Sorted array
   */
  const quickSort = (array) => {
    if (array.length < 2) return array; //базовый рекурсивный случай, когда в массиве 1 элемент
    const pivot = array[0]; //берем базовый элемент
    //убираем первый элемент массива и сортируем всех кто меньше
    const keysAreLessPivot = array.slice(1).filter((key) => key <= pivot);
    //убираем первый элемент массива и сортируем всех кто больше
    const keysAreMorePivot = array.slice(1).filter((key) => key > pivot);
    // массивы, который больше и меньше расставляем по краям от базового элемента, запускаем рекурсию
    return [
      ...quickSort(keysAreLessPivot),
      pivot,
      ...quickSort(keysAreMorePivot),
    ];
  };

  const myQuickSort = (array) => {
    if (array.length < 2) return array;
    const baseElement = array[0];

    const lessThenBase = array.slice(1).filter((item) => item < baseElement);
    const moreThanBase = array.slice(1).filter((item) => item > baseElement);

    return [
      ...myQuickSort(lessThenBase),
      baseElement,
      ...myQuickSort(moreThanBase),
    ];
  };

  console.log(myQuickSort([10, 5, 2, 3])); // [2, 3, 5, 10]
})();

// -----------------------------------------------------------------------------------
// with pointer https://habr.com/ru/companies/simbirsoft/articles/769312/
const partition = (arr, start, end) => {
  const pivot = arr[end]; // Определяем опорный элемент
  let i = start; // Определяем индекс, по которому делим массив на две части

  for (let j = start; j <= end - 1; j++) {
    if (arr[j] <= pivot) {
      [arr[i], arr[j]] = [arr[j], arr[i]]; // Меняем значения переменных
      i++;
    }
  }

  [arr[i], arr[end]] = [arr[end], arr[i]]; // Меняем значения переменных
  return i;
};

const quickSort = (arr, start, end) => {
  if (start < end) {
    // Условия запуска рекурсии
    const pi = partition(arr, start, end); // Получаем индекс

    quickSort(arr, start, pi - 1);
    quickSort(arr, pi + 1, end);
  }
};
