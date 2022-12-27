from pprint import pprint
from html.parser import HTMLParser
from collections import defaultdict

class SubstrateParser(HTMLParser):

    S_NONE = 0
    S_DETAILS = 1
    S_ERROR = 2
    S_LOCATION = 3

    IGNORE_TAGS = {'label', 'fieldset'}

    TRANSITION = {
        'Substrate Details': S_DETAILS,
        'Substrate Location History': S_LOCATION,
        'Substrate Error History': S_ERROR,
    }

    def __init__(self):
        super().__init__()
        self.text = list()
        self.status = self.S_NONE
        self.part = dict()

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORE_TAGS:
            return
        self.text = list()
        if tag == 'tr':
            self.row = list()
            self.col = 0
        elif tag == 'table':
            self.row = 0
        elif tag == 'unpublished': # Evatec Pisser können kein HTML
            self.text.append('<Unpublished>')

    def handle_data(self, data):
        self.text.append(data)

    def handle_endtag(self, tag):
        if tag in self.IGNORE_TAGS:
            return
        text = ''.join(self.text)
        if tag == 'td':
            self.row.append(text)
        if tag == 'tr':
            self.part[self.status].append(self.row.copy())
        elif tag == 'h2':
            s = TRANSITION.get(text)
            if s:
                self.status = s
                self.part[self.status] = list()

if __name__ == '__main__':
    tgc = SubstrateParser()

    fn = '../../tests/pto/PJ22 2021-07-22T10-19-03/SubstrateDetails01.html'

    with open(fn, 'rt') as fh:
        tgc.feed(fh.read())
    pprint(tgc.part)
