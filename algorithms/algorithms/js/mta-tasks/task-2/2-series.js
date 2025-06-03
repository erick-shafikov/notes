// Запустить таски последовательно, как только все таски выполнятся вернуть "done", ms время выполнения тасок
const tasks = [
    { task: () => console.log(1), ms: 2000 },
    { task: () => console.log(2), ms: 1000 },
    { task: () => console.log(3), ms: 4000 },
    { task: () => console.log(4), ms: 3000 },
  ];

  function series(tasks){
    //code
    console.log("done");
  }


  series(tasks);
// 2000ms => 1
// 2000ms + 1000ms => 2
// 2000ms + 1000ms + 4000ms => 3
// 2000ms + 1000ms + 4000ms + 3000ms => 4
// done