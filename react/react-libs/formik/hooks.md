<!-- useField -------------------------------------------------------------------------------------------------------------------------------->

# useField

```js
//FieldHookConfig
const config = {
  name: "field-name",
  value: "value",
  validate: (value) => {}, //undefined | string | Promise<any>
  type: "input-type",
  multiple: false,
  value: "someValue",
  // other input props
  ...config,
};

const [fieldProps, originalMeta, helpers] = useField(config);

// FieldInputProps
const fieldProps = {
  name: "field-name",
  onBlur: () => {},
  onChange: () => {},
  value: "",
};

// FieldMetaProps
const originalMeta = {
  error: "",
  initialError: undefined,
  initialTouched: false,
  initialValue: "",
  touched: false,
  value: "",
};

//FieldHelperProps;
const helpers = {
  setError: (value) => {},
  setTouched: (value, shouldValidate) => {},
  setValue: (value, shouldValidate) => {},
};
```

<!-- useFormik ------------------------------------------------------------------------------------------------------------------------------->

# useFormik()

для использования формы без контекста

Компоненты Field, FastField, ErrorMessage, connect(), and FieldArray не работают с useFormik

```jsx
const SignupForm = () => {
  const formik = useFormik({
    initialValues: {
      firstName: "",
      lastName: "",
      email: "",
    },
    onSubmit: (values) => {
      alert(JSON.stringify(values, null, 2));
    },
  });
  return (
    <form onSubmit={formik.handleSubmit}>
      <label htmlFor="firstName">First Name</label>
      <input
        id="firstName"
        name="firstName"
        type="text"
        onChange={formik.handleChange}
        value={formik.values.firstName}
      />
      <button type="submit">Submit</button>
    </form>
  );
};
```

# useFromContext()

```js
const context = useFromContext();

context = {
  dirty: false,
  errors: {},
  getFieldHelpers: (name) => {},
  getFieldMeta: (name) => {},
  getFieldProps: (nameOrOptions) => {},
  handleBlur: () => {},
  handleChange: () => {},
  handleReset: () => {},
  handleSubmit: () => {},
  initialErrors: {},
  initialStatus: undefined,
  initialTouched: {},
  initialValues: {},
  isSubmitting: false,
  isValid: true,
  isValidating: false,
  registerField: (name, _ref3) => {},
  resetForm: (nextState) => {},
  setErrors: (errors) => {},
  setFieldError: (field, value) => {},
  setFieldTouched: () => {},
  setFieldValue: () => {},
  setFormikState: (stateOrCb) => {},
  setStatus: (status) => {},
  setSubmitting: (isSubmitting) => {},
  setTouched: () => {},
  setValues: () => {},
  status: undefined,
  submitCount: 0,
  submitForm: () => {},
  touched: {},
  unregisterField: (name) => {},
  validateField: () => {},
  validateForm: () => {},
  validateOnBlur: true,
  validateOnChange: true,
  validateOnMount: false,
  values: { category: "", name: "", producer: "", price: "", metalType: "" },
};
```

<!-- withFormik ------------------------------------------------------------------------------------------------------------------------------>

# withFormik()

Hoc
