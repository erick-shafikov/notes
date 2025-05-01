//адаптер для сервиса
export interface ICrashReporterService {
  report(error: any): string;
}
