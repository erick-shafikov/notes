# React Native и Form компонент

```js
// react native
import { useForm, Form } from "react-hook-form";
function App() {
  const {
    control,
    register,
    formState: { isSubmitSuccessful, errors },
  } = useForm();

  return (
    <Form
      action="/api"
      control={control}
      render={({ submit }) => {
        <View>
          {isSubmitSuccessful && <Text>Form submit successful.</Text>}

          {errors?.root?.server && <Text>Form submit failed.</Text>}
          <Button onPress={() => submit()} />
        </View>;
      }}
    />
  );
}
```

# React Native и Controller компонент

```js
import { Text, View, TextInput, Button, Alert } from "react-native";
import { useForm, Controller } from "react-hook-form";

export default function App() {
  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues: {
      firstName: "",
      lastName: "",
    },
  });
  const onSubmit = (data) => console.log(data);

  return (
    <View>
      <Controller
        control={control}
        rules={{
          required: true,
        }}
        render={({ field: { onChange, onBlur, value } }) => (
          <TextInput
            placeholder="First name"
            onBlur={onBlur}
            onChangeText={onChange}
            value={value}
          />
        )}
        name="firstName"
      />
      {errors.firstName && <Text>This is required.</Text>}

      <Controller
        control={control}
        rules={{
          maxLength: 100,
        }}
        render={({ field: { onChange, onBlur, value } }) => (
          <TextInput
            placeholder="Last name"
            onBlur={onBlur}
            onChangeText={onChange}
            value={value}
          />
        )}
        name="lastName"
      />

      <Button title="Submit" onPress={handleSubmit(onSubmit)} />
    </View>
  );
}
```
