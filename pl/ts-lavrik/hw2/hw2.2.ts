export default {};

type TBaseUser = {
	id: number,
	email: string
}

type TAdminUser = TBaseUser & {
	type: 'admin',
	some: number
}

type TManagerUser = TBaseUser & {
	type: 'manager',
	accessLevel: number
}

type TClientUser = TBaseUser & {
	type: 'client',
	age: number,
	blocked: boolean
}

type TUser = TAdminUser | TManagerUser | TClientUser;

const users: TUser[] = [
	{ type: 'admin', id: 1, email: '1', some: 1,},
	{ type: 'manager', id: 2, email: '2', accessLevel: 1 },
	{ type: 'client', id: 1, email: '1', age: 18, blocked: false },
	{ type: 'client', id: 1, email: '1', age: 18, blocked: false }

]

// const adminUsers = users.filter((user: TUser) :user is Extract<TUser, {type: 'admin'}> => user.type  === 'admin');

//1------------------------------------------------------------------------------
/* const userFilter = <T>(type: T) => {
	return (item : TUser) : item is Extract<TUser, {type: T}> => item.type === type
};
const adminUsers1 = users.filter(userFilter('admin'));
console.log(adminUsers1); */
//------------------------------------------------------------------------------

//2-----------------------------------------------------------------------------
const typeFilter = <T extends {type: string}, U extends T['type']>(type : U) => {
	return (item: T) : item is Extract<T, {type: U}> => item.type === type
};

const adminUsers2 = users.filter(typeFilter('client'));
console.log(adminUsers2);

//3-----------------------------------------------------------------------------

// const typeFilter = <K extends string, V >(key: K, value : V) => {
// 	return (item : Record<K, V>) : item is Extract<typeof item, {key: V}> => item[key] === value
// };

// const adminUsers3 = users.filter(typeFilter('type', 'client'));