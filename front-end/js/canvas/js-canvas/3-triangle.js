let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');

// Треугольник

ctx.beginPath();
ctx.strokeStyle = 'red';
ctx.lineWidth = '4';
ctx.moveTo(50, 150);
ctx.lineTo(150, 50);
ctx.lineTo(200, 150);
// ctx.lineTo(50, 150); // замыкает по точкам
ctx.lineCap = 'round';
ctx.fillStyle = 'yellow';
ctx.closePath(); // замыкает автоматически
ctx.stroke();
ctx.fill();
