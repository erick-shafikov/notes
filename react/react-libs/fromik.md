# Хуки

## useField

```ts
const field: TUseFieldReturn = useField(props);

type TUseFieldProps =
  | string
  | {
      name: string;
      validate?: (value: any) => undefined | string | Promise<any>;
      type?: string; //text, number ...
      multiple?: boolean; //могут ли быть выбраны несколько
      value?: string;
    };

// возвращает массив
type TUseFieldReturn = [
  FieldInputProps<Value>,
  FieldMetaProps<Value>,
  FieldHelperProps
];

type FieldInputProps<Value> = {
  name: string;
  checked?: boolean;
  onBlur: () => void;
  onChange: (e: React.ChangeEvent<any>) => void;
  value: Value;
  multiple?: boolean;
};

type FieldMetaProps<Value> = {
  error: string;
  initialError: string;
  initialTouched: boolean;
  initialValue?: Value;
  touched: boolean;
  value: any;
};

type FieldHelperProps = {
  setValue(value: any, shouldValidate?: boolean): Promise<void | FormikErrors>;
  setTouched(value: boolean, shouldValidate?: boolean): void;
  setError(value: any): void;
};
```
