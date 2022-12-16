import sys, logging
from flask import Flask


app = Flask(__name__)

print(app.logger.handlers)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(levelname)s %(message)s')
handler.setFormatter(formatter)

app.logger.addHandler(handler)

app.logger.error('Error')

