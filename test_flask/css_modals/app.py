import re
import flask

app = flask.Flask(__name__)


class CSSModal():
    def __init__(self, id, html, compact=False):
        self.id = str(id)
        self.compact = compact
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', id):
            raise ValueError("ID '%s' contains invalid characters", self.id)
        self.html = html
        self.modal_id = 'modal_bg_' + self.id
        self.open_id = 'modal_open_' + self.id
        self.close_id = 'modal_close_' + self.id

    def js(self):
        js = '''
var {s.modal_id} = document.getElementById({s.modal_id});
var {s.open_id} = document.getElementById({s_open_id});
var {s.close_id} = document.getElementById({s_close_id});

{s.open_id}.onclick = function() {
  {modal_id}.style.display = "block";
}

{s.close_id}.onclick = function() {
  {s.modal_id}.style.display = "none";
}

/*
window.onclick = function(event) {
  if (event.target == {modal}) {
    {modal}.style.display = "none";
  }
}
*/
        '''\
        .format(s=self)
        return js

    def button(self):
        pass

    def div_modal(self):
        pass


@app.route('/')
def index():
    modal = CSSModal('abc', '')
    flask.render_template("index.html", modal=modal)

