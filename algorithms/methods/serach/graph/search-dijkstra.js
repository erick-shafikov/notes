/* 
        - 6 → a - 1 → 
начало         ↑3     конец 
        - 2 → b - 5 → 
*/

(function () {
  // описание графа
  const graph = {};
  graph.start = {};
  graph.start.a = 6;
  graph.start.b = 2;

  graph.a = {};
  graph.a.fin = 1;

  graph.b = {};
  graph.b.a = 3;
  graph.b.fin = 5;

  graph.fin = {};

  //Стоимость добраться до каждой от начала (вспомогательная таблица)
  const costs = {};
  costs.a = 6;
  costs.b = 2;
  costs.fin = Infinity; //нет прямого пути

  //Таблица родителей для каждого из узлов
  const parents = {};
  parents.a = "start";
  parents.b = "start";
  parents.fin = null; //не знаем в начальное точке как добраться

  // обработанные
  let processed = [];

  /**
   * Находит наименьший по значению среди соседей
   * @param {Object} itCosts Hash table
   * @returns {(string|null)} The lowest node
   */
  const findLowestCostNode = (itCosts) => {
    let lowestCost = Infinity;
    let lowestCostNode = null;

    Object.keys(itCosts).forEach((node) => {
      const cost = itCosts[node];
      // проходим по всем соседям
      if (cost < lowestCost && !processed.includes(node)) {
        //находим самого дешевого
        lowestCost = cost;
        lowestCostNode = node;
      }
    });
    return lowestCostNode;
  };

  let node = findLowestCostNode(costs);

  while (node !== null) {
    const cost = costs[node];
    //проходим по соседям
    const neighbors = graph[node];
    Object.keys(neighbors).forEach((n) => {
      const newCost = cost + neighbors[n];

      if (costs[n] > newCost) {
        // ... update the cost for this node
        costs[n] = newCost;
        // This node becomes the new parent for this neighbor.
        parents[n] = node;
      }
    });

    // Mark the node as processed
    processed.push(node);

    // Find the next node to process, and loop
    node = findLowestCostNode(costs);
  }

  console.log("Cost from the start to each node:");
  console.log(costs); // { a: 5, b: 2, fin: 6 }
})();
