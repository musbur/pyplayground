import io
from xml.etree.ElementTree import ElementTree, Element

et = ElementTree()
r = Element('root')
e = Element('abc')
e.text='This is me'
r.append(e)
et._setroot(r)

bb = io.BytesIO()
et.write(bb, encoding='ascii')

print(bb.getvalue())
