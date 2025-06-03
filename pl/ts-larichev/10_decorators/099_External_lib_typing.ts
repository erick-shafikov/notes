//'really-relaxed-json'  - библа без типизации
//вариант 1 выключить ts//@ts-ignore - выключение ts
// создать types.d.ts
import {toJson} from 'really-relaxed-json';

const rjson = '[ one two three {foo:bar} ]'
const json = toJson(rjson)