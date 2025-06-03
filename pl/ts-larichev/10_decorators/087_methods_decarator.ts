export {};

interface IUserService {
    users: number;
    getUsersInDatabase(): number;
}

class UserService implements IUserService {
    users: number = 1000;

    @LogFactory()
    @Log
    getUsersInDatabase(): number {
        throw new Error('Error');
    }
}
/* 
при создании декоратора метода в функцию перенаправляются 3 аргумента 
- target
- propertyKey
- descriptor {value: , writable, enumerable: boolean, configurable: boolean } value - сама функция, которую мы переопределяем ниже
*/
function Log(
    target: Object,
    propertyKey: string | symbol,
    descriptor: TypedPropertyDescriptor<(...args: any[]) => any>
): TypedPropertyDescriptor<(...args: any[]) => any> | void {
    const oldValue = descriptor.value; // для использования старого метода
    descriptor.value = () => {
        console.log('no error');
        // oldValue() можем использовать старую логику
    };
}

function LogFactory() {
    return (
        target: Object,
        propertyKey: string | symbol,
        descriptor: TypedPropertyDescriptor<(...args: any[]) => any>
    ): TypedPropertyDescriptor<(...args: any[]) => any> | void => {
        const oldValue = descriptor.value; // для использования старого метода
        descriptor.value = () => {
            console.log('no error');
            // oldValue() можем использовать старую логику
        };
    };
}

console.log(new UserService().getUsersInDatabase());
