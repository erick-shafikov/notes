export{}

interface IUserService {
    users: number;
    getUsersInDatabase() : number;
}

class UserService implements IUserService {
    @Max(100)//декоратор свойства
    users: number = 1000;

    getUsersInDatabase() : number {
        throw new Error('Error')
    }
}

function Max(max : number) {
    return (
        target: Object, // target - текущий ключ (user)
        propertyKey: string | symbol, //propertyKey - текущее значение 1000
        )  => {
            let value: number; //значение для замыканя getter
            const setter = function (newValue: number) { //логика setter
                if (newValue > max){
                    console.log(`enable to set value more than ${max}`)
                } else {
                    value = newValue;
                } 
            }

            const getter = function (){
                return value;
            }
//основная реализация здесь через defineProperty
            Object.defineProperty(target, propertyKey, { //определяем для ключа target логику
                get: getter,
                set: setter
            })
        }
}

console.log(new UserService().getUsersInDatabase())