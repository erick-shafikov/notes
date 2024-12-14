/* 
props:
  disabled - делает кнопку неактивной
  variant - text, contained, outlined
  onClick- для обработки click
  startIcon={<IconComponent />}> - для вставки иконки

Разновидности:
  IconButton
*/

import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import DeleteIcon from "@mui/icons-material/Delete";
import SendIcon from "@mui/icons-material/Send";
import IconButton from "@mui/material/IconButton";

import AlarmIcon from "@mui/icons-material/Alarm";
import AddShoppingCartIcon from "@mui/icons-material/AddShoppingCart";

import { styled } from "@mui/material/styles";

import CloudUploadIcon from "@mui/icons-material/CloudUpload";

//--------------------------------------------------------------------

export function BasicButtons() {
  return (
    <Stack spacing={2} direction="row">
      <Button variant="text">Text</Button>
      <Button variant="contained">Contained</Button>
      <Button variant="outlined">Outlined</Button>
    </Stack>
  );
}

//--------------------------------------------------------------------

export function TextButtons() {
  return (
    <Stack direction="row" spacing={2}>
      <Button>Primary</Button>

      <Button disabled>Disabled</Button>
      <Button href="#text-buttons">Link</Button>
    </Stack>
  );
}
//--------------------------------------------------------------------
export function ContainedButtons() {
  return (
    <Stack direction="row" spacing={2}>
      <Button variant="contained">Contained</Button>
      <Button variant="contained" disabled>
        Disabled
      </Button>
      <Button variant="contained" href="#contained-buttons">
        Link
      </Button>
    </Stack>
  );
}
//--------------------------------------------------------------------
export function IconLabelButtons() {
  return (
    <Stack direction="row" spacing={2}>
      <Button variant="outlined" startIcon={<DeleteIcon />}>
        Delete
      </Button>
      <Button variant="contained" endIcon={<SendIcon />}>
        Send
      </Button>
    </Stack>
  );
}
//--------------------------------------------------------------------

export function IconButtons() {
  return (
    <Stack direction="row" spacing={1}>
      <IconButton aria-label="delete" color="secondary">
        <DeleteIcon />
      </IconButton>
      <IconButton aria-label="delete" disabled color="primary">
        <DeleteIcon />
      </IconButton>
      <IconButton color="secondary" aria-label="add an alarm">
        <AlarmIcon />
      </IconButton>
      <IconButton color="primary" aria-label="add to shopping cart">
        <AddShoppingCartIcon />
      </IconButton>
    </Stack>
  );
}
//--------------------------------------------------------------------
// прячем под кнопку input, размером 1px
const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

export function InputFileUpload() {
  return (
    <Button
      component="label"
      variant="contained"
      startIcon={<CloudUploadIcon />}
    >
      Upload file
      <VisuallyHiddenInput type="file" />
    </Button>
  );
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
export default function Buttons() {
  return (
    <>
      {/* <BasicButtons /> */}
      {/* <TextButtons /> */}
      {/* <ContainedButtons /> */}
      {/* <IconLabelButtons /> */}
      {/* <IconButtons /> */}
      <InputFileUpload />
    </>
  );
}
