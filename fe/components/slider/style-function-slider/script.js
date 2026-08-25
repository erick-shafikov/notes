(function () {
  const slider = document.querySelector(".slider");

  const controlBtns = document.querySelectorAll(".control-btn");
  const prevBtn = document.querySelector(".control-btn-prev");
  const nextBtn = document.querySelector(".control-btn-next");

  let current = 5;
  const total = document.querySelectorAll(".slide-item").length;

  // Функция переключения
  function switchSlider(newIndex) {
    if (newIndex < 1) newIndex = 1;
    if (newIndex > total) newIndex = total;

    current = newIndex;

    updateSlider(current);
  }

  // Функция обновления
  function updateSlider(current) {
    slider.style.setProperty("--current", current);
  }

  // Клик по номеру
  controlBtns.forEach((link, index) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchSlider(index + 1);
    });
  });

  // Кнопки «Назад» и «Вперёд»
  if (prevBtn) {
    prevBtn.addEventListener("click", (e) => {
      e.preventDefault();
      switchSlider(current - 1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", (e) => {
      e.preventDefault();
      switchSlider(current + 1);
    });
  }

  // Синхронизируем состояние на старте с разметкой
  updateSlider(current);
})();
