/* 
    Шаблон определяет зависимость между объектами, чтобы при изменении состояния одного из них его «подчинённые» узнавали об этом.
    В шаблоне «Наблюдатель» есть объект («субъект»), ведущий список своих «подчинённых» («наблюдателей») 
    и автоматически уведомляющий их о любом изменении своего состояния, обычно с помощью вызова одного из их методов.

    Observer Определяет отношение между объектами "один-ко-многим", так что когда один объект изменяется, 
    все его дочерние объекты получают информацию об этом и автоматически обновляются

    Observer.Attach(ObjA);Observer.Attach(ObjB); 
    Observer.ChangeSomething(); 
    Observer.Notify();

*/

class JobPost {
  protected title: string;

  constructor(title: string) {
    this.title = title;
  }

  public getTitle() {
    return this.title;
  }
}

class JobSeeker {
  protected name: string;

  constructor(name: string) {
    this.name = name;
  }
  // метод для оповещения
  public onJobPosted(job: JobPost) {
    // Do something with the job posting
    console.log("Hi " + this.name + "! New job posted: " + job.getTitle());
  }
}

class JobPostings {
  protected observers: Array<JobSeeker>;
  // метод для включения функционала
  protected notify(jobPosting: JobPost) {
    this.observers.forEach((observer) => observer.onJobPosted(jobPosting));
  }

  public attach(observer: JobSeeker) {
    this.observers.push(observer);
  }

  public addJob(jobPosting: JobPost) {
    this.notify(jobPosting);
  }
}
// Создаём подписчиков
const johnDoe = new JobSeeker("John Doe");
const janeDoe = new JobSeeker("Jane Doe");

// Создаём публикатора и прикрепляем подписчиков
const jobPostings = new JobPostings();
jobPostings.attach(johnDoe);
jobPostings.attach(janeDoe);

// Добавляем новую вакансию и смотрим, будут ли уведомлены подписчики
jobPostings.addJob(new JobPost("Software Engineer"));

// Output
// Hi John Doe! New job posted: Software Engineer
// Hi Jane Doe! New job posted: Software Engineer
