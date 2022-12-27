import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
    url = flask.url_for('.other')
    print('URL:', url)
    return flask.make_response(
        '<a href="{url}">Click me</a>\n'.format(url=url))

@app.route('/other')
def other():
    return flask.make_response('It worked\n')

if __name__ == '__main__':
    app.run(port=8008, host='0.0.0.0')
