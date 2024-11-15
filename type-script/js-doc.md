# Definitions

## Function

```js
/**
 * @param {Merchant['id']} merchantId
 * @param {{
 *  name: string,
 *  url: string,
 *  categoryId: number,
 *  contactEmail: string,
 *  contactPhone: string,
 *  contactFullName: string,
 * }} formData
 * @return {Promise<Shop>} Newly created shop point info
 */
const addShopPoint: async (merchantId, formData) => {
  const { data } = await httpClient.post(
    `/merchants/${merchantId}/points`,
    formData
  );

  return data;
};
```

## Object

```js
/**
 * @typedef {{
 *  id: number;
 *  userId: number;
 *  name: string;
 *  address: string;
 *  fullName: string|null;
 *  email: string;
 *  phone: string|null;
 *  isPayout: boolean;
 * }} Merchant
 */
```

# Import. Импорт сущностей

С помощью объявления сущности внутри модуля

```js
/**
 * @typedef {import('_somePathToFile_').__entityName__} __SomeType__
 */
```

в дальнейшем этот тип можно импортировать в другом файле

# ReactComponent. Типизация пропсов React компонентов

```js
import React from "react";
import "./Button.css";

/**
 * A simple button component.
 *
 * @component
 * @param {Object} props - The component accepts text and onClick as props
 * @param {string} props.text - The text to be display on the button
 * @param {function} props.onClick - The click event handler.
 * @returns {JSX.Element} The rendered button component.
 *
 * @example
 * // Render a button with the text "Click Me"
 * <Button text="Click Me" onClick={() => console.log('Button clicked!')} />
 */
function Button({ text, onClick }) {
  return (
    <button className="myButton" onClick={onClick}>
      {text}
    </button>
  );
}

export default Button;
```
