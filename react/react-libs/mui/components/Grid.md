import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Grid from "@mui/material/Grid";
/* 
<Box /> - компонент для редактирования, пропы - elementType и sx
<Container /> -  горизонтально выравнивающий компонент fixed и fluid - пропы отвечающие за размер контейнера
<Grid container />
Два основных элемента
<Grid container 
  spacing={расстояние внутри}
  rowSpacing={1} columnSpacing={{ xs: 1, sm: 2, md: 3 }}
  />
  
  - spacing
<Grid
  item={элемент сетки}
  xs={от 1 до 12 колонок}
  xs="auto" - адаптивный вариант размера
  sm={адаптив}
  columns={{ xs: 4, sm: 8, md: 12 } вариант для определения колонок}/>
 */
const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: "center",
  color: theme.palette.text.secondary,
}));

export default function BasicGrid() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={2}>
        <Grid item xs={8}>
          <Item>xs=8</Item>
        </Grid>
        <Grid item xs={4}>
          <Item>xs=4</Item>
        </Grid>
        <Grid item xs={4}>
          <Item>xs=4</Item>
        </Grid>
        <Grid item xs={8}>
          <Item>xs=8</Item>
        </Grid>
      </Grid>
    </Box>
  );
}
