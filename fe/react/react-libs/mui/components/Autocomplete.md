```tsx
/*
props:  
  freeSolo === true //в поле может быть и произвольное значение не из списка опций
  selectOnFocus //при фокусе ставит автовыделение
  clearOnBlur //
  handleHomeEndKeys //при нажатии на клавиши home и end будет переносить в начало и конец списка соответственно
*/
import TextField from "@mui/material/TextField";
import Autocomplete, { autocompleteClasses } from "@mui/material/Autocomplete";
import { Fragment, useEffect, useState } from "react";
import {
  CountryType,
  FilmWithTitle,
  countries,
  options,
  top100FilmsWithLabel,
  top100FilmsWithTitle,
} from "./_mock-data";
import { styled, lighten, darken } from "@mui/system";
import {
  Box,
  CircularProgress,
  Stack,
  Theme,
  ThemeProvider,
  createTheme,
  useAutocomplete,
  useTheme,
} from "@mui/material";
import { sleep, timeSlots } from "./_utils";

//--------------------------------------------------------------------

//компонент по умолчанию
export function ComboBox() {
  return (
    <Autocomplete
      disablePortal={true}
      id="combo-box-demo"
      options={top100FilmsWithLabel}
      sx={{ width: 300 }}
      renderInput={(params) => <TextField {...params} label="Movie" />}
    />
  );
}
//--------------------------------------------------------------------

// пример контролируемого компонента
/* 
  the "value" state with the value/onChange props combination. 
  This state represents the value selected by the user, for instance when pressing Enter.
  the "input value" state with the inputValue/onInputChange props combination. 
  This state represents the value displayed in the textbox.
*/
export function ControllableStates() {
  const [value, setValue] = useState<string | null>(options[0]); //состояние компонента автозаполнения
  const [inputValue, setInputValue] = useState(""); //состояние компонента ControllableStates

  return (
    <div>
      <div>{`value: ${value !== null ? `'${value}'` : "null"}`}</div>
      <div>{`inputValue: '${inputValue}'`}</div>
      <br />
      <Autocomplete
        value={value} //значение для внешнего состояния
        onChange={(_event: unknown, newValue: string | null) => {
          setValue(newValue); //обработка внешнего состояния
        }}
        inputValue={inputValue}
        onInputChange={(_event, newInputValue) => {
          setInputValue(newInputValue);
        }}
        id="controllable-states-demo"
        options={options}
        sx={{ width: 300 }}
        renderInput={(params) => <TextField {...params} label="Controllable" />}
      />
    </div>
  );
}
//--------------------------------------------------------------------

// пример с группировкой элементов списка
const GroupHeader = styled("div")(({ theme }) => ({
  position: "sticky",
  top: "-8px",
  padding: "4px 10px",
  color: theme.palette.primary.main,
  backgroundColor:
    theme.palette.mode === "light"
      ? lighten(theme.palette.primary.light, 0.85)
      : darken(theme.palette.primary.main, 0.8),
}));

const GroupItems = styled("ul")({
  padding: 0,
});

export function RenderGroup() {
  const options = top100FilmsWithTitle.map((option) => {
    const firstLetter = option.title[0].toUpperCase();
    return {
      firstLetter: /[0-9]/.test(firstLetter) ? "0-9" : firstLetter,
      ...option,
    };
  });

  return (
    <Autocomplete
      id="grouped-demo"
      options={options.sort(
        (a, b) => -b.firstLetter.localeCompare(a.firstLetter)
      )}
      groupBy={(option) => option.firstLetter}
      getOptionLabel={(option) => option.title}
      sx={{ width: 300 }}
      renderInput={(params) => (
        <TextField {...params} label="With categories" />
      )}
      renderGroup={(params) => (
        <li key={params.key}>
          <GroupHeader>{params.group}</GroupHeader>
          <GroupItems>{params.children}</GroupItems>
        </li>
      )}
    />
  );
}
//--------------------------------------------------------------------

//c неактивными опциями выбора
export function DisabledOptions() {
  return (
    <Autocomplete
      id="disabled-options-demo"
      options={timeSlots}
      //функция возвращает true для деактивации опции списка
      getOptionDisabled={(option) =>
        option === timeSlots[0] || option === timeSlots[2]
      }
      sx={{ width: 300 }}
      renderInput={(params) => (
        <TextField {...params} label="Disabled options" />
      )}
    />
  );
}
//--------------------------------------------------------------------

//с использованием UseAutocomplete
const Label = styled("label")({
  display: "block",
});

const Input = styled("input")(({ theme }) => ({
  width: 200,
  backgroundColor: theme.palette.mode === "light" ? "#fff" : "#000",
  color: theme.palette.mode === "light" ? "#000" : "#fff",
}));

const Listbox = styled("ul")(({ theme }) => ({
  width: 200,
  margin: 0,
  padding: 0,
  zIndex: 1,
  position: "absolute",
  listStyle: "none",
  backgroundColor: theme.palette.mode === "light" ? "#fff" : "#000",
  overflow: "auto",
  maxHeight: 200,
  border: "1px solid rgba(0,0,0,.25)",
  "& li.Mui-focused": {
    backgroundColor: "#4a8df6",
    color: "white",
    cursor: "pointer",
  },
  "& li:active": {
    backgroundColor: "#2977f5",
    color: "white",
  },
}));

export function UseAutocomplete() {
  const {
    getRootProps,
    getInputLabelProps,
    getInputProps,
    getListboxProps,
    getOptionProps,
    groupedOptions,
  } = useAutocomplete({
    id: "use-autocomplete-demo",
    options: top100FilmsWithTitle,
    getOptionLabel: (option) => option.title,
  });

  return (
    <div>
      <div {...getRootProps()}>
        <Label {...getInputLabelProps()}>useAutocomplete</Label>
        <Input {...getInputProps()} />
      </div>
      {groupedOptions.length > 0 ? (
        <Listbox {...getListboxProps()}>
          {(groupedOptions as typeof top100FilmsWithTitle).map(
            (option, index) => (
              <li {...getOptionProps({ option, index })}>{option.title}</li>
            )
          )}
        </Listbox>
      ) : null}
    </div>
  );
}
//--------------------------------------------------------------------
// пример с асинхронной загрузкой данных
export function Asynchronous() {
  const [open, setOpen] = useState(false);
  const [options, setOptions] = useState<readonly FilmWithTitle[]>([]);
  const loading = open && options.length === 0;

  useEffect(() => {
    let active = true;

    if (!loading) {
      return undefined;
    }

    (async () => {
      await sleep(1e3); // For demo purposes.

      if (active) {
        setOptions([...top100FilmsWithTitle]);
      }
    })();

    return () => {
      active = false;
    };
  }, [loading]);

  useEffect(() => {
    if (!open) {
      setOptions([]);
    }
  }, [open]);

  return (
    <Autocomplete
      id="asynchronous-demo"
      sx={{ width: 300 }}
      open={open} //флаг на открытие
      onOpen={() => {
        setOpen(true); //cb на открытие
      }}
      onClose={() => {
        setOpen(false); //cb на закрытие
      }}
      isOptionEqualToValue={(option, value) => option.title === value.title}
      getOptionLabel={(option) => option.title}
      options={options}
      loading={loading}
      renderInput={(params) => (
        <TextField
          {...params}
          label="Asynchronous"
          InputProps={{
            ...params.InputProps,
            endAdornment: (
              <Fragment>
                {loading ? (
                  <CircularProgress color="inherit" size={20} />
                ) : null}
                {params.InputProps.endAdornment}
              </Fragment>
            ),
          }}
        />
      )}
    />
  );
}
//--------------------------------------------------------------------
//с выбором нескольких вариантов
export function LimitTags() {
  return (
    <Autocomplete
      multiple //флаг на множественный выбор
      limitTags={2}
      id="multiple-limit-tags"
      options={top100FilmsWithTitle}
      getOptionLabel={(option) => option.title}
      defaultValue={[
        top100FilmsWithTitle[13],
        top100FilmsWithTitle[12],
        top100FilmsWithTitle[11],
      ]}
      renderInput={(params) => (
        <TextField {...params} label="limitTags" placeholder="Favorites" />
      )}
      sx={{ width: "500px" }}
    />
  );
}

//--------------------------------------------------------------------
//кастомный компонент
export function CustomInputAutocomplete() {
  return (
    <label>
      Value:{" "}
      <Autocomplete
        sx={{
          display: "inline-block",
          "& input": {
            width: 200,
            bgcolor: "background.paper",
            color: (theme) =>
              theme.palette.getContrastText(theme.palette.background.paper),
          },
        }}
        id="custom-input-demo"
        options={options}
        renderInput={(params) => (
          <div ref={params.InputProps.ref}>
            {/* обязательно передавать ref */}
            <input type="text" {...params.inputProps} />
          </div>
        )}
      />
    </label>
  );
}

//--------------------------------------------------------------------
//кастомный компонент

// Theme.ts
const customTheme = (outerTheme: Theme) =>
  createTheme({
    palette: {
      mode: outerTheme.palette.mode,
    },
    components: {
      //проп для формирования кастомных компонентов
      MuiAutocomplete: {
        //а именно
        defaultProps: {
          renderOption: (props, option, _state, ownerState) => (
            <Box
              sx={{
                borderRadius: "8px",
                margin: "5px",
                //для одной опции
                [`&.${autocompleteClasses.option}`]: {
                  padding: "8px",
                },
              }}
              component="li"
              {...props}
            >
              {ownerState.getOptionLabel(option)}
            </Box>
          ),
        },
      },
    },
  });

export function GloballyCustomizedOptions() {
  // useTheme is used to determine the dark or light mode of the docs to maintain the Autocomplete component default styles.
  const outerTheme = useTheme();

  return (
    <ThemeProvider theme={customTheme(outerTheme)}>
      <Stack spacing={5} sx={{ width: 300 }}>
        <MovieSelect />
        <CountrySelect />
      </Stack>
    </ThemeProvider>
  );
}

function MovieSelect() {
  return (
    <Autocomplete
      options={top100FilmsWithTitle}
      // проп для формирования label
      getOptionLabel={(option: FilmWithTitle) =>
        `${option.title} (${option.year})`
      }
      id="movie-customized-option-demo"
      disableCloseOnSelect
      renderInput={(params) => (
        <TextField {...params} label="Choose a movie" variant="standard" />
      )}
    />
  );
}

function CountrySelect() {
  return (
    <Autocomplete
      id="country-customized-option-demo"
      options={countries}
      disableCloseOnSelect
      getOptionLabel={(option: CountryType) =>
        `${option.label} (${option.code}) +${option.phone}`
      }
      renderInput={(params) => (
        <TextField {...params} label="Choose a country" variant="filled" />
      )}
    />
  );
}
```
