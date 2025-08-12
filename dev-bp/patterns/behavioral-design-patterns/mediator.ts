/* 
Шаблон «Посредник» подразумевает добавление стороннего объекта («посредника») для управления взаимодействием между двумя объектами 
(«коллегами»). Шаблон помогает уменьшить связанность (coupling) классов, общающихся друг с другом, 
ведь теперь они не должны знать о реализациях своих собеседников.
Шаблон определяет объект, который инкапсулирует способ взаимодействия набора объектов.

Mediator Определяет объект, который инкапсулирует в себе набор объектов, которые могут взаимодействовать. 
Mediator.Add(ObjA); Mediator.Add(ObjB); ObjA.Send(“ObjB”, "Message");
*/

interface ChatRoomMediator {
  showMessage(user: User, message: string): void;
}

// Посредник принимает экземпляр класса, выполняя с ним функционал
class ChatRoom implements ChatRoomMediator {
  public showMessage(user: User, message: string) {
    const time = new Date();
    const sender = user.getName();

    console.log(time + `[${sender}]` + message);
  }
}

//каждый экземпляр получает посредника
class User {
  protected name: string;
  protected chatMediator: ChatRoomMediator;

  constructor(name: string, chatMediator: ChatRoomMediator) {
    this.name = name;
    this.chatMediator = chatMediator;
  }

  public getName() {
    return this.name;
  }

  //функционал
  public send(message: string) {
    this.chatMediator.showMessage(this, message);
  }
}

//Посредник
const mediator = new ChatRoom();

const john = new User("John Doe", mediator);
const jane = new User("Jane Doe", mediator);

john.send("Hi there!");
jane.send("Hey!");
