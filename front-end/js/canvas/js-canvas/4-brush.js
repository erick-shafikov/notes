let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');

// кисть

let myColor = 'red';
document.getElementById('color').oninput = function () {
  myColor = this.value;
};

canvas.onmousedown = (e) => {
  canvas.onmousemove = (e) => {
    let x = e.offsetX;
    let y = e.offsetY;
    ctx.fillRect(x - 5, y - 5, 10, 10);
    ctx.fillStyle = myColor;
    ctx.fill();
  };

  canvas.onmouseup = (e) => {
    canvas.onmousemove = null;
  };
};
