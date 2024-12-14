// сложность log2(n)

(function () {
  /**
   * Searches recursively number from the list
   * @param {Array} list упорядоченный список
   * @param {number} item элемент, который нужно найти
   * @returns {(number|null)} позиция, на который находится элемент
   */
  function binarySearch(list, item) {
    let low = 0; //стартовый индекс
    let high = list.length - 1; //индекс последнего элемент массива

    while (low <= high) {
      const mid = Math.floor((low + high) / 2); //берется середина
      const guess = list[mid]; //предполагаемый элемент посередине

      if (guess === item) {
        return mid; //угадали
      } else if (guess > item) {
        //если предполагаемый элемент меньше взятого из середины, то индекс конца массива приравниваем к этому элементу
        // так как искомый явно находится в меньшей половине
        high = mid - 1;
      } else {
        // если элемент оказался больше предполагаемого, то нижнюю граница сдвигаем до индекса предполагаемого элемента
        low = mid + 1;
      }
    }

    return null;
  }

  function myB(list, item) {
    let low = 0;
    let hight = list.length - 1;

    while (low < hight) {
      let guessIndex = Math.floor((low + hight) / 2);
      let guessElement = list[guessIndex];

      if (guessElement === item) {
        return guessIndex;
      } else if (guessElement > item) {
        hight = guessIndex;
      } else {
        low = guessIndex;
      }
    }

    return null;
  }

  const my_list = [1, 3, 5, 7, 9];

  console.log(myB(my_list, 3)); // 1
  console.log(myB(my_list, -1)); // null
})();
