// aka bubble sort
// -------------------------------------------------------------------
//grokking algorithms
(function () {
  /**
   * Находит наименьший элемент массива
   * @param {Array} array массив неупорядоченных значений
   * @returns {number} возвращает индекс наименьшего
   */
  const findSmallestIndex = (array) => {
    let smallestElement = array[0]; // пусть наименьший элемент - первый
    let smallestIndex = 0; // индекс наименьшего элемента

    for (let i = 1; i < array.length; i++) {
      if (array[i] < smallestElement) {
        //если элемент текущей итерации меньше наименьшего
        smallestElement = array[i]; //сохраняем этот элемент и его индекс
        smallestIndex = i;
      }
    }

    return smallestIndex;
  };

  /**
   *
   * @param {Array} array входной массив
   * @returns {Array} отсортированный массив
   */
  const selectionSort = (array) => {
    const sortedArray = []; //результирующий массив
    const copyArray = [...array]; //массив мутаций

    for (let i = 0; i < array.length; i++) {
      // находим в промежуточном массиве наименьшее значение
      const smallestIndex = findSmallestIndex(copyArray);
      // добавляем в итоговый массив элемент, который будет убран из промежуточного массива
      // splice удаляет элементы  и возвращает массив удаленных элементов,
      // в данном случае это наименьший элемент массива по индексу 0
      sortedArray.push(copyArray.splice(smallestIndex, 1)[0]);
    }

    return sortedArray;
  };

  const myFindSmallestIndex = (array) => {
    let smallestIndex = 0;
    let smallestElement = array[0];

    for (let i = 0; i < array.length; i++) {
      if (array[i] < smallestElement) {
        smallestElement = array[i];
        smallestIndex = i;
      }
    }

    return smallestIndex;
  };

  const mySelectionSort = (array) => {
    const arrayCopy = [...array];
    const resultArray = [];

    for (let i = 0; i < array.length; i++) {
      let smallestIndex = myFindSmallestIndex(arrayCopy);
      resultArray.push(arrayCopy.splice(smallestIndex, 1)[0]);
    }

    return resultArray;
  };

  const sourceArray = [5, 3, 6, 2, 10];
  const sortedArray = mySelectionSort([5, 3, 6, 2, 10]);

  console.log("Source array - ", sourceArray); // [5, 3, 6, 2, 10]
  console.log("New sorted array - ", sortedArray); // [2, 3, 5, 6, 10]
})();
// -------------------------------------------------------------------

// 2nd implementation https://habr.com/ru/companies/simbirsoft/articles/769312/
const bubbleSort = (arr) => {
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - i; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]]; // Меняем значения переменных
      }
    }
  }
};
