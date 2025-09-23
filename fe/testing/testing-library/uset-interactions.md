# setup

Функция позволяет настроить взаимодействие с браузером

```js
import userEvent from "@testing-library/user-event";

const user = userEvent.setup();

await user.keyboard("[ShiftLeft>]"); // Press Shift (without releasing it)
await user.click(element); // Perform a click with `shiftKey: true`
```

```ts
type TOptions = {
  advanceTimers: AdvancedTimers;
  applyAccept: boolean = true;
  autoModify: boolean = true;
  delay: number = 0;
  document: globalThis.document;
  keyboardMap: StandardKeyBoard;
  pointerEventsCheck:
    | PointerEventsCheckLevel.Never
    | PointerEventsCheckLevel.EachTarget
    | PointerEventsCheckLevel.EachApiCall
    | PointerEventsCheckLevel.EachTrigger;
  pointerMap: any;
  skipAutoClose: () => void | boolean = false;
  skipClick: () => void | boolean = false;
  skipHover: () => void | boolean = false;
  writeToClipboard: () => void | boolean = false;
};
```

желательно пользоваться userEvent.setup(), хотя все методы доступны и из userEvent

# pointer

```ts
function pointer(
  input: PointerActionInput | Array<PointerActionInput>
): Promise<void>;
```

```js
pointer({ keys: "[MouseLeft]" });
pointer({ keys: "[MouseLeft][MouseRight]" });
// or
pointer("[MouseLeft][MouseRight]");
pointer("[MouseLeft>]"); // press the left mouse button
pointer("[/MouseLeft]"); // release the left mouse button
// движение
pointer([
  // touch the screen at element1
  { keys: "[TouchA>]", target: element1 },
  // move the touch pointer to element2
  { pointerName: "TouchA", target: element2 },
  // release the touch pointer at the last position (element2)
  { keys: "[/TouchA]" },
]);
```

# Keyboard

```ts
function keyboard(input: KeyboardInput): Promise<void>;
```

```js
keyboard("foo"); // translates to: f, o, o
keyboard("{{a[["); // translates to: {, a, [
keyboard("{Shift}{f}{o}{o}"); // translates to: Shift, f, o, o
keyboard("{\\}}"); // translates to: }
keyboard("[ShiftLeft][KeyF][KeyO][KeyO]"); // translates to: Shift, f, o, o
keyboard("{a>}"); // press a without releasing it
keyboard("{a>5}"); // press a without releasing it and trigger 5 keydown
keyboard("{a>5/}"); // press a for 5 keydown and then release it
keyboard("{/a}"); // release a previously pressed a
```

# Clipboard

```ts
function copy(): Promise<DataTransfer | undefined>;
function cut(): Promise<DataTransfer | undefined>;
function paste(clipboardData?: DataTransfer | string): Promise<void>;
```

# clear()

```ts
clear(element: Element): Promise<void>
```

```js
test("clear", async () => {
  const user = userEvent.setup();

  render(<textarea defaultValue="Hello, World!" />);

  await user.clear(screen.getByRole("textbox"));

  expect(screen.getByRole("textbox")).toHaveValue("");
});
```

# selectOptions(), deselectOptions()

```ts
function selectOptions(
  element: Element,
  values: HTMLElement | HTMLElement[] | string[] | string,
): Promise<void>
deselectOptions(
  element: Element,
  values: HTMLElement | HTMLElement[] | string[] | string,
): Promise<void>
```

```js
test("selectOptions", async () => {
  const user = userEvent.setup();

  render(
    <select multiple>
      <option value="1">A</option>
      <option value="2">B</option>
      <option value="3">C</option>
    </select>
  );

  await user.selectOptions(screen.getByRole("listbox"), ["1", "C"]);

  expect(screen.getByRole("option", { name: "A" }).selected).toBe(true);
  expect(screen.getByRole("option", { name: "B" }).selected).toBe(false);
  expect(screen.getByRole("option", { name: "C" }).selected).toBe(true);
});

test("deselectOptions", async () => {
  const user = userEvent.setup();

  render(
    <select multiple>
      <option value="1">A</option>
      <option value="2" selected>
        B
      </option>
      <option value="3">C</option>
    </select>
  );

  await user.deselectOptions(screen.getByRole("listbox"), "2");

  expect(screen.getByText("B").selected).toBe(false);
});
```

# type()

```ts
type(
  element: Element,
  text: KeyboardInput,
  options?: {
    skipClick?: boolean
    skipAutoClose?: boolean
    initialSelectionStart?: number
    initialSelectionEnd?: number
  }
): Promise<void>
```

```js
test("type into an input field", async () => {
  const user = userEvent.setup();

  render(<input defaultValue="Hello," />);
  const input = screen.getByRole("textbox");

  await user.type(input, " World!");

  expect(input).toHaveValue("Hello, World!");
});
```

# upload()

```ts
function upload(
  element: HTMLElement,
  fileOrFiles: File | File[]
): Promise<void>;
```

```js
test("upload file", async () => {
  const user = userEvent.setup();

  render(
    <div>
      <label htmlFor="file-uploader">Upload file:</label>
      <input id="file-uploader" type="file" />
    </div>
  );
  const file = new File(["hello"], "hello.png", { type: "image/png" });
  const input = screen.getByLabelText(/upload file/i);

  await user.upload(input, file);

  expect(input.files[0]).toBe(file);
  expect(input.files.item(0)).toBe(file);
  expect(input.files).toHaveLength(1);
});

test("upload multiple files", async () => {
  const user = userEvent.setup();

  render(
    <div>
      <label htmlFor="file-uploader">Upload file:</label>
      <input id="file-uploader" type="file" multiple />
    </div>
  );
  const files = [
    new File(["hello"], "hello.png", { type: "image/png" }),
    new File(["there"], "there.png", { type: "image/png" }),
  ];
  const input = screen.getByLabelText(/upload file/i);

  await user.upload(input, files);

  expect(input.files).toHaveLength(2);
  expect(input.files[0]).toBe(files[0]);
  expect(input.files[1]).toBe(files[1]);
});
```

# click()

```ts
function click(element: Element, {ctrlKey: boolean, shiftKey: boolean}, {skipPointerEventsCheck: boolean}): Promise<void>;
function pointer([{target: element}, {keys: '[MouseLeft]', target: element}])
```

```js
test("click", async () => {
  const onChange = jest.fn();
  const user = userEvent.setup();

  render(<input type="checkbox" onChange={onChange} />);

  const checkbox = screen.getByRole("checkbox");

  await user.click(checkbox);

  expect(onChange).toHaveBeenCalledTimes(1);
  expect(checkbox).toBeChecked();
});
```

# dblClick()

```ts
function dblClick(element: Element): Promise<void>;
function pointer([
  { target: element },
  { keys: "[MouseLeft][MouseLeft]", target: element },
]);
```

```js
test("double click", async () => {
  const onChange = jest.fn();
  const user = userEvent.setup();

  render(<input type="checkbox" onChange={onChange} />);

  const checkbox = screen.getByRole("checkbox");

  await user.dblClick(checkbox);

  expect(onChange).toHaveBeenCalledTimes(2);
  expect(checkbox).not.toBeChecked();
});
```

# tripleClick()

```ts
function tripleClick(element: Element): Promise<void>;

pointer([
  { target: element },
  { keys: "[MouseLeft][MouseLeft][MouseLeft]", target: element },
]);
```

```js
test("triple click", async () => {
  const onChange = jest.fn();
  const user = userEvent.setup();

  render(<input type="checkbox" onChange={onChange} />);

  const checkbox = screen.getByRole("checkbox");

  await user.tripleClick(checkbox);

  expect(onChange).toHaveBeenCalledTimes(3);
  expect(checkbox).toBeChecked();
});
```

# hover(), unhover()

```ts
function hover(element: Element): Promise<void>;

pointer({ target: element });
```

```js
test("hover/unhover", async () => {
  const user = userEvent.setup();
  render(<div>Hover</div>);

  const hoverBox = screen.getByText("Hover");
  let isHover = false;

  hoverBox.addEventListener("mouseover", () => {
    isHover = true;
  });
  hoverBox.addEventListener("mouseout", () => {
    isHover = false;
  });

  expect(isHover).toBeFalsy();

  await user.hover(hoverBox);

  expect(isHover).toBeTruthy();

  await user.unhover(hoverBox);

  expect(isHover).toBeFalsy();
});
```

# tab()

```ts
function tab(options: { shift?: boolean }): Promise<void>;
```

```js
// without shift
keyboard("{Tab}");
// with shift=true
keyboard("{Shift>}{Tab}{/Shift}");
// with shift=false
keyboard("[/ShiftLeft][/ShiftRight]{Tab}");
```

```js
test("tab", async () => {
  const user = userEvent.setup();
  render(
    <div>
      <input type="checkbox" />
      <input type="radio" />
      <input type="number" />
    </div>
  );

  const checkbox = screen.getByRole("checkbox");
  const radio = screen.getByRole("radio");
  const number = screen.getByRole("spinbutton");

  expect(document.body).toHaveFocus();

  await user.tab();

  expect(checkbox).toHaveFocus();

  await user.tab();

  expect(radio).toHaveFocus();

  await user.tab();

  expect(number).toHaveFocus();

  await user.tab();

  // cycle goes back to the body element
  expect(document.body).toHaveFocus();

  // simulate Shift-Tab
  await user.tab({ shift: true });

  expect(number).toHaveFocus();
});
```
