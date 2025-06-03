export default {};

type TBaseUser = {
	id: number,
	email: string
}

type TManagerUser = TBaseUser & {
  type: 'manager',
  accessLevel: number
}

type TAdminUser = TBaseUser & {
	type: 'admin',
	some: number
}


type TClientUser = TBaseUser & {
	type: 'client',
	age: number,
	blocked: boolean
}


type CommonPart1<obj1 extends object, obj2 extends object> = {[X in keyof obj1] : X extends keyof obj2? obj1[X] : never} //общие - один из, разные - never
// type CommonPart2<obj1, obj2> = {[X in keyof (obj1 | obj2)] : X extends keyof obj2 ? obj2[X] : never}//общие - один из, разные - нет
type CommonPart2<obj1 extends object, obj2 extends object> = {
	[K in keyof obj1 as (K extends keyof obj2 ? K : never)] : obj1[K] 
}
type CommonPart3<obj1, obj2> = Pick<
obj1,
{
	[K in keyof obj1 & keyof obj2]: obj1[K] extends obj2[K]
		? obj2[K] extends obj1[K]
			? K
			: never
		: never;
}[keyof obj1 & keyof obj2]
>//общие - только общие
type CommonPart4<obj1, obj2> = Omit<obj1 | obj2, keyof Omit<obj1, keyof obj2>> //общие - объединить, разные - нет

type T1 = CommonPart1<TAdminUser, TClientUser>
type T2 = CommonPart2<TAdminUser, TClientUser>
type T3 = CommonPart3<TAdminUser, TClientUser>
type T4 = CommonPart4<TAdminUser, TClientUser>