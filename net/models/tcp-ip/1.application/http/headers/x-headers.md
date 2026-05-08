Заголовки с префиксом X- — это исторически неофициальные / кастомные HTTP-заголовки. если заголовок стал популярным — убирают X-. Если делать запрос fetch то браузер делает pre-flight потому что это уже “non-simple header”. префикс X- считается legacy-подходом

# X-Content-Type-Options (res)

Определяет MIME тип указанные в [Content-type](./res-headers.md#content-type) цель - повышение безопасности, то есть если в ответе один тип данных а в заголовке другой

# X-DNS-Prefetch-Control (res)

нестандартный, для пред-загрузки контента

```bash
X-DNS-Prefetch-Control: on
X-DNS-Prefetch-Control: off
```

```html
<meta http-equiv="x-dns-prefetch-control" content="off" />
<link rel="dns-prefetch" href="https://www.mozilla.org" />
<link rel="dns-prefetch" href="//www.mozilla.org" />
```

# X-Forwarded-For (req)

для идентификации исходного IP-адреса клиента, подключающегося к веб-серверу через прокси-сервер

```bash
# X-Forwarded-For: <client>, <proxy>
# X-Forwarded-For: <client>, <proxy>, …, <proxyN>

X-Forwarded-For: 2001:db8:85a3:8d3:1319:8a2e:370:7348
X-Forwarded-For: 203.0.113.195
X-Forwarded-For: 203.0.113.195, 2001:db8:85a3:8d3:1319:8a2e:370:7348
```

# X-Forwarded-Host (req)

для идентификации исходного хоста, запрошенного клиентом в заголовке HTTP-запроса Host. Стандартизированной версией этого заголовка является заголовок HTTP [Forwarded](./req-headers.md#forwarded), хотя он используется гораздо реже.

# X-Forwarded-Proto (req)

для идентификации протокола (HTTP или HTTPS), который клиент использовал для подключения к прокси-серверу или балансировщику нагрузки. Стандартизированной версией этого заголовка является заголовок HTTP [Forwarded](./req-headers.md#forwarded), хотя он используется гораздо реже.

# X-Frame-Options (res)

Заголовок может использоваться для указания того, следует ли разрешить браузеру отображать документ в frame, iframe, embed или object. Если этот заголовок не отправлен, и веб-сайт не реализовал никаких других механизмов ограничения встраивания (например, директиву CSP frame-ancestors), то браузер разрешит другим сайтам встраивать этот документ.

```bash
X-Frame-Options: DENY
X-Frame-Options: SAMEORIGIN
```

[Content-Security-Policy](./res-headers.md#content-security-policy) используется директива frame-ancestors, которую следует использовать вместо этой.

# X-Permitted-Cross-Domain-Policies (res)

определяет метаполитику, которая контролирует, может ли документ, работающий в веб-клиенте, таком как Adobe Acrobat или Microsoft Silverlight, предоставлять доступ к ресурсам сайта из других источников

# X-Powered-By (res)

для идентификации приложения или фреймворка, сгенерировавшего ответ.

# X-Robots-Tag (res)

Поисковые краулеры используют правила из заголовка X-Robots-Tag для настройки способа отображения веб-страниц или других ресурсов в результатах поиска.

```bash
# X-Robots-Tag: <indexing-rule>
# X-Robots-Tag: <indexing-rule>, …, <indexing-ruleN>
# X-Robots-Tag: <indexing-rule>, <bot-name>: <indexing-rule>
# X-Robots-Tag: <bot-name>: <indexing-rule>, …, <indexing-ruleN>
```

indexing-rule:

- all - Нет ограничений на индексацию или отображение в результатах поиска
- noindex - Не отображать эту страницу, медиафайл или ресурс в результатах поиска
- nofollow - Не переходите по ссылкам на этой странице
- none = noindex + nofollow
- nosnippet - Не отображайте фрагмент текста или предварительный просмотр видео в результатах поиска для этой страницы
- indexifembedded - Поисковой системе разрешается индексировать содержимое страницы, если оно встроено в другую страницу с помощью iframe или аналогичных HTML-элементов, несмотря на правило noindex. Правило indexifembedded имеет эффект только в том случае, если оно сопровождается правилом noindex.
- max-snippet: number - В результатах поиска используйте текстовый фрагмент, содержащий максимум number символов
- max-image-preview:
- - none
- - standard
- - large
- max-video-preview: 0, -1
- notranslate
- noimageindex
- unavailable_after: date/time

# X-XSS-Protection (res) (dep)

Заголовок ответа HTTP X-XSS-Protection был функцией Internet Explorer, Chrome и Safari, которая предотвращала загрузку страниц при обнаружении отраженных атак межсайтового скриптинга (XSS). Рекомендуется использовать [Content-Security-Policy](./res-headers.md#content-security-policy)

```bash
X-XSS-Protection: 0 # фильтр выключен
X-XSS-Protection: 1 # если есть нарушение, то браузер удалит небезопасные части страницы
X-XSS-Protection: 1; mode=block # предотвращение на стадии рендеринга
X-XSS-Protection: 1; report=<reporting-uri> # отправить отчет по  report-uri
```
