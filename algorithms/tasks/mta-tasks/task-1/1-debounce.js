// Написать debounce
function debounce(ms){
    // code
  }
  

const sum = debounce(((a, b) => {
    console.log(a + b);
  }),1000)
  
  sum(1, 1); // не выполнится
  sum(1, 2); // не выполнится
  sum(1, 3); // не выполнится
  sum(1, 4); // 5