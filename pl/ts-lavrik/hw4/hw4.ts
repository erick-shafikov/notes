type TStringSplitter<S extends string, D extends string> =  
S extends `${infer P1}${D}${infer P2}` ? [P1, ...TStringSplitter<P2, D>] : [S]

type TCamelKeysString<S, D extends string> =
S extends `${infer P1}${D}${infer P2}` ? `${Capitalize<P1>}${Capitalize<TCamelKeysString<P2, D>>}` : S

type a = TStringSplitter<'hello.world.where', '.'>
type b = TStringSplitter<'some_backend_key', '_'>
type c = TCamelKeysString<'hello.world.where', '.'>

function splitStr(str: string, delim: string){}

let parts = splitStr('hello.world.where', '.')

type TCamelKeysObj<T extends Record<string, any>> = {
  [K in keyof T as TCamelKeysString<K, '_'>] : T[K]
}

type TSnakeObj = {
  'key_1': any,
  'key_2': any,
  'key_3': any,
  'another_long_key': any
}

type TCamelObjFromSnakeObj = TCamelKeysObj<TSnakeObj>