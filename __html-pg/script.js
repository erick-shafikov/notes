console.log(1);

setTimeout(() => console.log(2), 0);

new Promise((resolve) => {
  console.log(3);
  resolve(4);
}).then((res) => console.log(res));

console.log(5);
