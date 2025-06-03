// Запустить таски паралельно, как только все таски выполнятся вернуть "done", ms время выполнения тасок
const tasks = [
    {task: ()=> console.log(1), ms: 2000},
    {task: ()=> console.log(2), ms: 1000},
    {task: ()=> console.log(3), ms: 4000},
    {task: ()=> console.log(4), ms: 300},
  ];

  function paraller(tasks){
    //code
    console.log("done");
  }

  paraller(tasks); 
// 300ms => 4
// 1000ms => 2
// 2000ms => 1
// 4000ms => 3
// done