function deleteListItem() {
  const listItem = this.parentNode;

  // Still need JS to animate out?
  // Need to inert this out?
  listItem.classList.add("is-deleting");
  setTimeout(() => {
    listItem.parentNode.removeChild(listItem);
  }, 500);
}

function addListItem() {
  // Get the input value from the text field
  const inputField = document.getElementById("newItem");
  const newItemText = "New Item";

  // Create a new list item element
  const newItem = document.createElement("li");
  newItem.className = "item";

  // Create a paragraph element with the new item text
  const itemText = document.createElement("p");
  itemText.textContent = newItemText;
  newItem.appendChild(itemText);

  // Create a delete button element
  const deleteButton = document.createElement("button");
  deleteButton.className = "delete";
  deleteButton.addEventListener("click", deleteListItem);

  // Add an X symbol to the delete button
  const xSymbol = document.createElement("span");
  xSymbol.setAttribute("aria-hidden", "true");
  xSymbol.textContent = "❌";
  deleteButton.appendChild(xSymbol);

  // Add a visually hidden label to the delete button
  const srLabel = document.createElement("span");
  srLabel.className = "sr-only";
  srLabel.textContent = "delete";
  deleteButton.appendChild(srLabel);

  // Add the delete button to the list item element
  newItem.appendChild(deleteButton);

  // Add the new list item to the list
  const list = document.querySelector(".items");
  list.appendChild(newItem);
}

const addButton = document.querySelector(".add");
const deleteButtons = document.querySelectorAll(".delete");

addButton.addEventListener("click", addListItem);
deleteButtons.forEach((button) => {
  button.addEventListener("click", deleteListItem);
});
