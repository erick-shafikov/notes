(function () {
  const graph = {};
  graph.you = ['alice', 'bob', 'claire'];
  graph.bob = ['anuj', 'peggy'];
  graph.alice = ['peggy'];
  graph.claire = ['thom', 'jonny'];
  graph.anuj = [];
  graph.peggy = [];
  graph.thom = [];
  graph.jonny = [];

  /**
   * Функция, которая определяет кто продает манго (тот у кого последняя буква m)
   * @param {string} name строка - имя
   * @returns {boolean} Result of checking
   */
  const personIsSeller = (name) => name[name.length - 1] === 'm';

  /**
   * Find a mango seller
   * @param {string} name На входе имя
   * @returns {boolean} Search results
   */
  const search = (name) => {
    // разворачиваем всю ветку дерева по определенному имени в массив имен друзей
    let searchQueue = [...graph[name]];
    // массив для уже проверенных элементов дерева
    const searched = [];
    // пока в ветке поиска есть элементы
    while (searchQueue.length) {
      //берем первый элемент из массива и удаляем его
      const person = searchQueue.shift();
      //если его еще не было в очереди
      if (searched.indexOf(person) === -1) {
        //проверяем он ли продавец манго
        if (personIsSeller(person)) {
          console.log(`${person} is a mango seller!`);
          //выходим из while-цикла
          return true;
        }
        // если он не продавец манго, то добавляем в очередь поиска все поддерево этого элемента
        searchQueue = searchQueue.concat(graph[person]);
        // добавляем в очередь
        searched.push(person);
      }
    }
    return false;
  };

  const mySearch = (name) => {
    let subTree = [...graph[name]];
    const checkedPersons = [];

    while (subTree.length) {
      const potential = subTree.shift();

      if (checkedPersons.indexOf(potential) === -1) {
        if (personIsSeller(potential)) {
          console.log(potential);
          return true;
        }
        subTree = subTree.concat(graph[potential]);
        checkedPersons.push(potential);
      }
    }
    return false;
  };

  mySearch('you'); // thom is a mango seller!
})();
