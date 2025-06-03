let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');
let x = 0;
let timer;

function drawSin() {
  y = 30 * Math.sin(x) + 100;

  if (x >= 400) {
    x = 0;
  } else {
    x = x + 0.3;
  }

  ctx.fillRect(5 * x, 20 * y + 100, 2, 2);
  timer = setTimeout(drawSin, 10);
}

drawSin();
