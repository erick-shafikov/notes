// создать функцию построения таблицы по схеме
const cols = [
    {id: "name", value: "name"},
    {id: "sname", value: "secondName"},
    {id: "phone", value: "phone"}
  ];
  
  const data = [
    {name: "Vasya", sname: "Sokolov", phone: "89221231234"},
    {name: "Ivan", sname: "Petrov", phone: "89326783293"},
    {name: "Kate", sname: "Pan", phone: "89386783493"},
  ]
  
  function createTable(scheme, data){
    //code
  }
  
  console.log(createTable(cols, data));
  /** [
  [ 'name', 'secondName', 'phone' ],
  [ 'Vasya', 'Sokolov', '89221231234' ],
  [ 'Ivan', 'Petrov', '89326783293' ],
  [ 'Kate', 'Pan', '89386783493' ]
] */