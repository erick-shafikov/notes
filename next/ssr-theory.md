ssr - страница формируется на стороне сервера и отдается готовая клиенту
ssg - страница формируется на стадии билда приложения
csr - страница формируется на стороне клиента

- streaming - страница формируется на сервере. на странице оставляются placeholder для каждого элемента, но подгружается по частям с помощью suspense

rsc - react server components нет пере-рендера, нет хуков без клиентской логикой. Нужно учитывать позицию в дереве компонентов, если родительский компонент серверный, то все дочерние тоже клиентские
rcc - react client component если клиентская логика.

rsc payload - результат рендеринга, места для клиентских, данные