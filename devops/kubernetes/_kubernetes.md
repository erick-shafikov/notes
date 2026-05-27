# Kubernetes

Смысл — оркестрация контейнерами (Docker и другими runtime), позволяет "подменять" упавший контейнер (Pod) на его реплику и делать это быстро.

## Архитектура кластера

### Мастер-нода (Control Plane)

- **kube-apiserver** — REST API сервер, единственная точка входа для управления кластером (frontend Kubernetes)
- **kube-scheduler** — подбирает подходящую Worker-ноду для нового Pod-а исходя из ресурсов и ограничений
- **kube-controller-manager** — группа контроллеров:
  - node controller — мониторинг нод
  - replication controller — восстановление нужного числа реплик
  - endpoint controller — связывает Service и Pod
  - service account & token controller
- **etcd** — key-value хранилище, сохраняет всё состояние кластера

> `kubectl` — это CLI-инструмент для работы с кластером, не компонент мастер-ноды.

### Worker-нода

- **kubelet** — агент на каждой ноде, слушает команды от kube-apiserver, тянет образы и запускает Pod-ы
- **kube-proxy** — управляет сетевыми правилами (iptables/IPVS), обеспечивает связь между Pod-ами и Service-ами
- **container runtime** — среда выполнения контейнеров (Docker, containerd, CRI-O)
- **Pod** — атомарная единица, изолирующая один или несколько контейнеров

---

## kubectl — основные команды

```bash
# Просмотр ресурсов
kubectl get pods
kubectl get pods -n namespace_name        # в конкретном namespace
kubectl get pods -o wide                  # с IP и нодой
kubectl get all                           # все ресурсы
kubectl get deployments
kubectl get services
kubectl get nodes

# Подробности
kubectl describe pod pod_name
kubectl describe deployment deploy_name

# Логи
kubectl logs pod_name
kubectl logs pod_name -c container_name   # если несколько контейнеров
kubectl logs -f pod_name                  # follow

# Выполнить команду в контейнере
kubectl exec -it pod_name -- /bin/sh

# Применить манифест
kubectl apply -f file.yaml

# Удалить ресурс
kubectl delete pod pod_name
kubectl delete -f file.yaml

# Масштабирование
kubectl scale deployment deploy_name --replicas=3

# Rollout
kubectl rollout status deployment/deploy_name
kubectl rollout undo deployment/deploy_name   # откат

# Port forwarding
kubectl port-forward pod/pod_name 8080:80
```

---

## Pod

Pod — обёртка над контейнером (или группой тесно связанных контейнеров), атомарная единица в Kubernetes. Контейнеры внутри одного Pod-а делят сеть и storage.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: default
  labels:
    app: my-app
spec:
  containers:
    - name: mysql
      image: mysql:5.6
      ports:
        - containerPort: 3306
      env:
        - name: MYSQL_ROOT_PASSWORD
          value: secret
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
```

---

## Namespaces

Способ логической изоляции ресурсов внутри одного кластера. Удобно для разделения окружений (dev/staging/prod) или команд.

```bash
kubectl get namespaces
kubectl create namespace my-ns
kubectl apply -f file.yaml -n my-ns
```

По умолчанию три namespace: `default`, `kube-system`, `kube-public`.

---

## Labels и Selectors

Labels — произвольные key-value метки на ресурсах. Selectors — способ находить ресурсы по лейблам. Именно так Service и Deployment связываются с Pod-ами.

```yaml
metadata:
  labels:
    app: frontend
    env: production
```

```bash
kubectl get pods -l app=frontend
kubectl get pods -l env=production,app=frontend
```

---

## Deployment

Управляет набором одинаковых Pod-ов (через ReplicaSet). Обеспечивает rolling update, откат версий, масштабирование.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  strategy:
    type: RollingUpdate # или Recreate
    rollingUpdate:
      maxSurge: 1 # сколько Pod-ов сверх нормы при обновлении
      maxUnavailable: 1 # сколько Pod-ов могут быть недоступны
  template: # шаблон Pod-а
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: nginx:1.25
          ports:
            - containerPort: 80
```

---

## ReplicaSet

Гарантирует, что в кластере работает заданное число реплик Pod-а. Deployment управляет ReplicaSet автоматически — напрямую ReplicaSet создают редко.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: frontend-rs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: nginx:1.25
```

---

## Service

Стабильная точка доступа к Pod-ам (у Pod-ов IP меняются при пересоздании). Service выбирает Pod-ы по лейблам через selector.

### Типы Service

| Тип            | Описание                                              |
| -------------- | ----------------------------------------------------- |
| `ClusterIP`    | Доступен только внутри кластера (по умолчанию)        |
| `NodePort`     | Открывает порт на каждой ноде (30000–32767)           |
| `LoadBalancer` | Создаёт внешний балансировщик (облако: AWS/GCP/Azure) |
| `ExternalName` | DNS-алиас на внешний хост                             |

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
spec:
  type: ClusterIP # или NodePort, LoadBalancer
  selector:
    app: frontend # находит Pod-ы с таким лейблом
  ports:
    - protocol: TCP
      port: 80 # порт Service
      targetPort: 8080 # порт контейнера
```

NodePort пример:

```yaml
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080 # внешний порт (опционально, иначе выберется автоматически)
```

---

## Ingress

Управляет внешним HTTP/HTTPS доступом к сервисам кластера. Требует установленного Ingress Controller (nginx, Traefik и др.).

```text
Интернет → Ingress → Service → Deployment → Pods
```

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: frontend-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # nginx.ingress.kubernetes.io/rewrite-target: /$1
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - example.com
      secretName: example-com-tls # Secret с TLS-сертификатом
  rules:
    - host: example.com
      http:
        paths:
          - path: /
            pathType: Prefix # Prefix | Exact | ImplementationSpecific
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 8080
```

Популярные Ingress Controllers: NGINX, Traefik, HAProxy, Istio.

```bash
kubectl get ingress
kubectl describe ingress frontend-ingress
kubectl apply -f ingress.yaml
```

---

## ConfigMap

Хранит незасекреченные конфигурационные данные (переменные окружения, конфиг-файлы).

```bash
kubectl create configmap db-config --from-literal=DB_HOST=localhost --from-literal=DB_PORT=5432
```

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: db-config
data:
  DB_HOST: localhost
  DB_PORT: "5432"
```

Использование в Pod:

```yaml
spec:
  containers:
    - name: app
      image: myapp:1.0
      # Все переменные из ConfigMap сразу
      envFrom:
        - configMapRef:
            name: db-config
      # Или отдельная переменная
      env:
        - name: DATABASE_HOST
          valueFrom:
            configMapKeyRef:
              name: db-config
              key: DB_HOST
```

---

## Secrets

Хранит чувствительные данные (пароли, токены, сертификаты). Данные хранятся в base64-кодировке (не шифрование — только encoding).

```bash
kubectl create secret generic db-secret --from-literal=DB_PASSWORD=mysecretpassword
```

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  DB_PASSWORD: bXlzZWNyZXRwYXNzd29yZA== # base64: echo -n 'mysecretpassword' | base64
```

Использование в Pod:

```yaml
spec:
  containers:
    - name: app
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: DB_PASSWORD
```

---

## Volumes, PersistentVolume, PVC

**Volume** — подключаемое хранилище для Pod-а. В отличие от контейнерной файловой системы, данные не теряются при перезапуске контейнера (но могут теряться при удалении Pod-а).

### emptyDir — временное хранилище (живёт пока жив Pod)

```yaml
spec:
  containers:
    - name: app
      volumeMounts:
        - mountPath: /tmp/data
          name: temp-storage
  volumes:
    - name: temp-storage
      emptyDir: {}
```

### PersistentVolume (PV) — ресурс хранилища в кластере

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce # RWO: одна нода | ReadOnlyMany | ReadWriteMany
  persistentVolumeReclaimPolicy: Retain # или Delete, Recycle
  hostPath:
    path: /mnt/data # для локальной разработки
```

### PersistentVolumeClaim (PVC) — запрос на хранилище от Pod-а

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
```

Подключение PVC к Pod:

```yaml
spec:
  containers:
    - name: app
      volumeMounts:
        - mountPath: /var/data
          name: my-storage
  volumes:
    - name: my-storage
      persistentVolumeClaim:
        claimName: my-pvc
```

---

## Liveness и Readiness Probes

Probes — проверки состояния контейнера.

- **livenessProbe** — если падает, контейнер перезапускается
- **readinessProbe** — если падает, Pod убирается из Service (трафик не идёт)
- **startupProbe** — даёт время на старт (блокирует liveness/readiness до готовности)

```yaml
spec:
  containers:
    - name: app
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 10
        periodSeconds: 5
        failureThreshold: 3
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 3
      # TCP probe пример:
      # tcpSocket:
      #   port: 3306
      # Exec probe пример:
      # exec:
      #   command: ["mysqladmin", "ping"]
```

---

## StatefulSet

Как Deployment, но для stateful-приложений (БД, Kafka, ZooKeeper). Гарантирует:

- стабильные имена Pod-ов (`pod-0`, `pod-1`, ...)
- стабильные PVC для каждого Pod-а
- упорядоченный старт и остановку

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql # headless service
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
        - name: mysql
          image: mysql:8.0
          env:
            - name: MYSQL_ROOT_PASSWORD
              value: secret
          volumeMounts:
            - name: data
              mountPath: /var/lib/mysql
  volumeClaimTemplates: # PVC создаётся для каждого Pod-а отдельно
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

---

## DaemonSet

Запускает ровно один Pod на каждой ноде кластера. Используется для мониторинга, логирования, сетевых агентов.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: log-collector
spec:
  selector:
    matchLabels:
      app: log-collector
  template:
    metadata:
      labels:
        app: log-collector
    spec:
      containers:
        - name: fluentd
          image: fluentd:latest
```

---

## Job и CronJob

**Job** — запускает Pod до успешного завершения (batch-задачи).

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  completions: 1
  parallelism: 1
  template:
    spec:
      restartPolicy: Never # OnFailure или Never (не Always!)
      containers:
        - name: migration
          image: myapp:1.0
          command: ["python", "migrate.py"]
```

**CronJob** — Job по расписанию (cron-синтаксис).

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-job
spec:
  schedule: "0 2 * * *" # каждый день в 02:00
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: cleanup
              image: myapp:1.0
              command: ["python", "cleanup.py"]
```

---

## HorizontalPodAutoscaler (HPA)

Автоматически масштабирует число реплик Deployment на основе метрик (CPU, память, кастомные метрики).

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70 # масштабировать при >70% CPU
```

```bash
kubectl get hpa
kubectl describe hpa frontend-hpa
```

---

## Типичная связка объектов

```text
[Внешний трафик]
       ↓
   Ingress            ← HTTP/HTTPS роутинг
       ↓
   Service            ← стабильный IP, балансировка
       ↓
  Deployment          ← управление версиями, rolling update
       ↓
  ReplicaSet          ← поддержание числа реплик
       ↓
    Pods              ← запущенные контейнеры
       ↓
  ConfigMap/Secret    ← конфигурация и секреты
  PVC → PV            ← персистентное хранилище
```
