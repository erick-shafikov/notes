let canvas = document.getElementById('c1');
let ctx = canvas.getContext('2d');

//Линии

ctx.beginPath(); //объявление нового отрезка
ctx.strokeStyle = 'red'; //цвет линии
ctx.lineWidth = '5'; //ширина линии
ctx.moveTo(100, 50); //начало курсора
ctx.lineTo(150, 150); //рисуем линию до
ctx.stroke(); //обводим, иначе не видно

ctx.beginPath();
ctx.strokeStyle = 'blue'; //цвет линии
ctx.lineWidth = '20'; //ширина линии
ctx.moveTo(200, 50); //новое начало курсора
ctx.lineTo(300, 50); //рисуем линию до
ctx.lineTo(300, 100); //рисуем линию до

ctx.lineCap = 'round'; //закругление концов
ctx.lineCap = 'square'; //квадрат концов
ctx.lineCap = 'butt'; //по умолчанию

ctx.stroke(); //обводим, иначе не видно
ctx.clearRect(0, 0, 400, 200);
