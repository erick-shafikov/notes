let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');

//круги

const pi = Math.PI;
ctx.beginPath();
ctx.lineWidth = 5;
ctx.strokeStyle = 'red';
ctx.fillStyle = 'yellow';
ctx.arc(150, 100, 75, pi / 2, false); //центр окружности
// ctx.arc(X, Y, R, pi / 2 - Длина дуги, false - против часовой); //центр окружности
ctx.stroke();
ctx.fill();

ctx.beginPath();
ctx.lineWidth = 5;
ctx.strokeStyle = 'green';
ctx.fillStyle = 'pink';
ctx.arc(300, 100, 75, pi / 2, false); //центр окружности
// ctx.arc(X, Y, R, pi / 2 - Длина дуги, false - против часовой); //центр окружности
ctx.stroke();
ctx.fill();
ctx.clearRect(0, 0, 400, 200);

canvas.onmousemove = (e) => {
  let x = e.offsetX;
  let y = e.offsetY;

  ctx.beginPath();
  ctx.clearRect(0, 0, 400, 200);
  let radius = Math.pow(Math.pow(x - 200, 2) + Math.pow(y - 100, 2), 0, 5);
  ctx.arc(200, 100, radius, 0, 2 * pi, false);
  ctx.stroke();
  ctx.fill();
};
