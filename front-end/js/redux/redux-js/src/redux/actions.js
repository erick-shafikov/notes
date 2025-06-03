//каждый из actions возвращает стрококвое представление типо 'INCREMENT',....
import { CHANGE_THEME, DECREMENT, DISABLE_BUTTINS, ENABLE_BUTTONS, INCREMENT, INCREMENT_BY_AMOUNT } from "./types"

export function increment(){
    return {
        type: INCREMENT
    }
}

export function decrement(){
    return {
        type: DECREMENT
    }
}

export function incrementByAmount(newValue){
    return {
        type: INCREMENT_BY_AMOUNT,
        payload: Number(newValue)
    }
}

export function enableButtons(){
    return {
        type: ENABLE_BUTTONS
    }
}

export function disableButtons(){
    return {
        type: DISABLE_BUTTINS
    }
}

export function changeTheme(newTheme){
    return {
        type: CHANGE_THEME,
        payload: newTheme
    }
}
//асинхронные - отдельный функции, внутри которых обычные reducer'ы
export function asyncIncrement(){
    return function(dispatch){
        dispatch(disableButtons())
        setTimeout(()=>{
            dispatch(increment())
            dispatch(enableButtons())
        }, 3000)
    }
}