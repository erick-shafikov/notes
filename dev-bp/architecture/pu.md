❏ app

├─❏ application (interfaces)
│ ├─❏ services
│ │ ├─chat (interface)
│ │ └─notification (interface)  
│ └─services.ts (interface)
│
├─❏ domains
│ ├─api.ts (interface)
│ ├─config.ts (interface)
│ ├─errors.ts (classes)
│ ├─sentry.ts (interface)
│ ├─storage.ts (interface)
│ ├─types.ts (utility types: PaginatedList...)
│ └─utils.ts
│
├─❏ features
│ ├─❏ application
│ │ └─❏ use-cases (services) => result
│ ├─❏ domain
│ │ ├─errors (class)
│ │ ├─repositories (interface)
│ │ ├─constants
│ │ ├─utils (interface)
│ │ └─types.ts
│ ├─❏ infrastructure
│ │ ├─api-some-repository.ts (implements domain layer)
│ │ └─fake-api-some-repository.ts (implements domain layer)
│ └─❏ ui
│ ` `├─❏ components
│ ` `├─❏ query-options
│ ` `└─❏ types
│
├─❏ infrastructure (implements domains/application services layer)
│ ├─config.ts
│ ├─api.ts
│ ├─chat.ts
│ ├─fake-chat.ts
│ ├─sentry.ts
│ ├─notification.ts
│ ├─storage.ts
│ └─fake-storage.ts
│
├─❏ routes
│ └...
│
└─❏ ui
` `├❏ components
` `├❏ hooks
` `└❏ utils
