const root = document.getElementById("field");

function main() {
  const fieldSizeX = 80;
  const fieldSizeY = 20;

  let ballX = 40;
  let ballY = 10;

  let moveX = 1;
  let moveY = 1;

  let playerAPosition = 20;
  let playerBPosition = 20;
  let rocketSize = 4;

  function draw() {
    let field = "";
    for (let rowIndex = 0; rowIndex < fieldSizeY; rowIndex++) {
      if (rowIndex === 0) {
        field += "┌" + "─".repeat(fieldSizeX - 2) + "┐" + "\n";
      } else if (rowIndex === fieldSizeY - 1) {
        field += "└" + "─".repeat(fieldSizeX - 2) + "┘" + "\n";
      } else if (rowIndex === ballY) {
        if (ballX === 0) {
          field += "X" + " ".repeat(fieldSizeX - 2) + "│" + "\n";
        } else if (ballX === fieldSizeX - 1) {
          field += "│" + " ".repeat(fieldSizeX - 2) + "X" + "\n";
        } else {
          field +=
            "│" +
            " ".repeat(ballX - 1) +
            "o" +
            " ".repeat(fieldSizeX - ballX - 2) +
            "│" +
            "\n";
        }
      }
      {
        field += "│" + " ".repeat(fieldSizeX - 2) + "│" + "\n";
      }
    }

    console.log(field);
  }

  function changeCords() {
    if (ballX === 0) {
      moveX = 1;
    }
    if (ballX === fieldSizeX - 1) {
      moveX = -1;
    }
    if (ballY === 0) {
      moveY = 1;
    }
    if (ballY === fieldSizeY - 1) {
      moveY = -1;
    }

    ballX = ballX + moveX;
    ballY = ballY + moveY;
  }

  setInterval(() => {
    changeCords();

    draw();
  }, 100);
}

window.addEventListener("keydown", (e) => {
  if (e.key === "a") {
    if (playerAPosition <= fieldSizeY - size) {
      playerAPosition += 1;
    }
  }
  if (e.key === "z") {
    if (playerAPosition > 0) {
      playerAPosition -= 1;
    }
  }
});

main();
