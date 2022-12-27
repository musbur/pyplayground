from werkzeug.wrappers import Request, Response
from werkzeug.routing import Map, Rule
from werkzeug.utils import escape

url_map = Map([
    Rule('/', endpoint='index'),
    Rule('/<x>', endpoint='something')
    ])

def wsgi_app(environ, start_response):
    request = Request(environ)
    adapter = url_map.bind_to_environ(request.environ)

    print('Adapter: ',  adapter.match())

    for k, v in request.args.items():
        print('%s: %s' % (k, v))

    response = Response(escape('Hello <"> World\n'))
    return response(environ, start_response)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('127.0.0.1', 8008, wsgi_app, use_debugger=True, use_reloader=True)
