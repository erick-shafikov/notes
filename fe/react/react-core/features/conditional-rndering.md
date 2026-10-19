# conditional rendering

Условный рендеринг

```tsx
function UserGreeting(): JSX.Element {
  return <h1>Welcome back!</h1>;
}
function GuestGreeting(): JSX.Element {
  return <h1>Please sign up.</h1>;
}

interface GreetingProps {
  isLoggedIn: boolean;
}

function Greeting({ isLoggedIn }: GreetingProps): JSX.Element {
  if (isLoggedIn) {
    return <UserGreeting />;
  }
  return <GuestGreeting />;
}
ReactDOM.render(
  // Try changing to isLoggedIn={true}:
  <Greeting isLoggedIn={true} />,
  document.getElementById("root")
);
```
