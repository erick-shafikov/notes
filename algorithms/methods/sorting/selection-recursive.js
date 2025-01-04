(function () {
  // поиск наименьшего индекса
  const findSmallest = (arr) => {
    let smallest = arr[0];
    let smallestIndex = 0;
    let arrLen = arr.length;

    for (let i = 0; i < arrLen; i++) {
      if (arr[i] < smallest) {
        smallest = arr[i];
        smallestIndex = i;
      }
    }
    return smallestIndex;
  };

  /**
   * Рекурсивный вариант
   * @param {Array} arr An array of numbers
   * @return {Array} New sorted array
   */
  const selectionSort = (arr) => {
    //базовый случай рекурсии
    if (!arr.length) return [];
    //найдем наименьший элемент в массиве, splice изменяет массив выбросив найденный элемент
    let smallest = arr.splice(findSmallest(arr), 1);
    //соединим найденный элемент и рекурсивный проход по ИЗМЕНЕННОМУ массиву arr уже без наименьшего элемента
    return smallest.concat(selectionSort(arr));
  };

  const mySelectionSort = (arr) => {
    if (!arr.length) return [];
    const arrayWithSmallest = arr.splice(findSmallest(arr, 1), 1);
    return arrayWithSmallest.concat(mySelectionSort(arr));
  };

  let arr = [5, 3, 6, 2, 10];

  console.log(mySelectionSort(arr));
})();
