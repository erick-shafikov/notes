export {}

interface IUserService {
    users: number;
    getUsersInDatabase() : number;
}

class UserService implements IUserService {
    users: number;
    
    private _users: number = 1000;


    getUsersInDatabase() : number {
        return this._users
    }

    setUsersInDatabase(@Positive() num: number) : void {
        this._users = num
    }
}
//дескриптор на парметры
function Positive() {
    return (
        target: Object, //UserService
        propertyKey: string | symbol,//setUsersInDatabase
        parameterIndex: number// Индекс параметра
        )  => {
            
        }
}

console.log(new UserService().getUsersInDatabase())