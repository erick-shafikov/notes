```html
<div class="container">
  <ul class="items">
    <li class="item">
      <p>Item 1 Name</p>
      <button class="delete">
        <span aria-hidden="true">❌</span>
        <span class="sr-only">delete</span>
      </button>
    </li>
    <li class="item">
      <p>Item 2 Name</p>
      <button class="delete">
        <span aria-hidden="true">❌</span>
        <span class="sr-only">delete</span>
      </button>
    </li>
    <li class="item">
      <p>Item 3 Name</p>
      <button class="delete">
        <span aria-hidden="true">❌</span>
        <span class="sr-only">delete</span>
      </button>
    </li>
    <li class="item">
      <p>Item 4 Name</p>
      <button class="delete">
        <span aria-hidden="true">❌</span>
        <span class="sr-only">delete</span>
      </button>
    </li>
  </ul>

  <button class="add">
    <span aria-hidden="true">➕</span>
    <span class="sr-only">add new item</span>
  </button>
</div>
```

```scss
.item {
  opacity: 1;
  height: 3rem;
  display: grid;
  overflow: hidden;
  transform-origin: bottom;
  transition: opacity 0.5s, transform 0.5s, height 0.5s, display 0.5s
      allow-discrete;
}

@starting-style {
  .item {
    opacity: 0;
    height: 0;
  }
}

/* while it is deleting, before DOM removal in JS */
.is-deleting {
  opacity: 0;
  height: 0;
  display: none;
  transform: skewX(50deg) translateX(-25vw);
}

/* etc */
@layer base {
  body {
    font-family: system-ui, sans-serif;
  }

  button {
    border: none;
    background: none;
  }

  .items {
    padding: 0;
    display: grid;
    gap: 0.5rem;
  }

  .item {
    display: grid;
    grid-template-columns: 1fr auto;
    background: aliceblue;
    border: 1px solid lightblue;
    padding: 0 1rem;
    border-radius: 1rem;
    width: 300px;
  }

  .sr-only {
    clip: rect(0 0 0 0);
    clip-path: inset(50%);
    height: 1px;
    overflow: hidden;
    position: absolute;
    white-space: nowrap;
    width: 1px;
  }

  .container {
    display: grid;
    justify-content: center;
    width: 300px;
    margin: 0 auto;
  }

  .add {
    font-size: 2rem;
    width: 3rem;
    height: 3rem;
    line-height: 0;
    margin: 0 auto;
  }
}
```

```js
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
```
