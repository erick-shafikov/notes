let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');

// движущаяся точка
let x = 200;
let y = 100;
let stepCount = 0;
let direction;
let timer;
let myX;
let myY;

function drawDot() {
  ctx.clearRect(0, 0, 400, 200);

  if (stepCount === 0) {
    stepCount = Math.floor(15 * Math.random());
    direction = Math.floor(8 * Math.random());
  } else {
    stepCount--;
  }

  switch (direction) {
    case 0:
      y = y - 1;
      break;
    case 1:
      x = x + 1;
      break;
    case 2:
      y = y + 1;
      break;
    case 3:
      x = x - 1;
      break;
    case 4:
      x = x + 1;
      y = y - 1;
    case 5:
      x = x + 1;
      y = y + 1;
      break;
    case 6:
      x = x - 1;
      y = y + 1;
      break;
    case 7:
      x = x - 1;
      y = y - 1;
      break;
    default:
      break;
  }

  if (x < 0 || x > 400 || y < 0 || y > 200) step = 0;

  ctx.fillRect(x - 3, y - 3, 6, 6);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(myX, myY);
  ctx.stroke();

  timer = setTimeout(drawDot, 100);
}

drawDot();

canvas.onmousemove = (e) => {
  myX = e.offsetX;
  myY = e.offsetY;
};
