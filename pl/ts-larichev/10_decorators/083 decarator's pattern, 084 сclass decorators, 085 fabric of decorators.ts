function setUserAdvanced(users: number) {
    //фабрика декораторов
    return <T extends { new (...args: any[]): {} }>(constructor: T) => {
        //более конкретная типизация
        return class extends constructor {
            users = users;
        };
    };
}

function nullUser(target: Function) {
    //типизация через функцию, target - это класс
    target.prototype.users = 0;
}

//фабрика декораторов
function setUsers(users: number) {
    return (target: Function) => {
        target.prototype.users = users;
    };
}

function threeUserAdvanced<T extends { new (...args: any[]): {} }>(
    constructor: T
) {
    //более конкретная типизация
    return class extends constructor {
        users = 3;
    };
}
//сервис по получения user
interface IUserService {
    users: number;
    getUsersInDataBase(): number;
}

// @nullUser - такой декоратор изменяет только на стадии инициализации экземпляра класса
@setUserAdvanced(4)
@setUsers(2)
@threeUserAdvanced
class UserService implements IUserService {
    users: number = 1000;

    getUsersInDataBase(): number {
        return this.users;
    }
}

console.log(new UserService().getUsersInDataBase());

export {};
