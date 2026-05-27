# Виртуализация

Структура машины:

- Hardware (host os)
- Hypervisor - позволяет создавать VM разделяют по типам: железо, программная, включает в себя:
- ├─VM1:
  │ ├─VM
  │ ├─Windows (guest os)
  │ └─Application
  ├─VM2:
  │ ├─VM
  │ ├─Windows (guest os)
  │ └─Application
  ...

snapshot - состояние VM в определенный момент, для отката
cluster - несколько Hypervisor
