"""
Hi all,

I would like to modify the standard str.format() in a way that when the
input field is of type str, there is some character replacement, and the
string gets padded or truncated to the given field width. Basically like
this:

fmt = MagicString('<{s:6}>')
print(fmt.format(s='Äußerst'))

Output:
<Aeusse>

I've written a function fix_format() which, given a string and a field
width, does just that. However, I find myself unable to implement a
Formatter that uses this function in the intened way. See the example below,
I hope I sprinkled it with enough comments to make my intents clear.
"""

### Self contained example
import re
from string import Formatter

_replacements = [(re.compile(rx), repl) for rx, repl in (\
    ('Ä', 'Ae'),
    ('ä', 'ae'),
    ('Ö', 'Oe'),
    ('ö', 'oe'),
    ('Ü', 'Ue'),
    ('ü', 'ue'),
    ('ß', 'ss'))]

def fix_format(text, width):

    # Seven regex passes seems awfully inefficient. I can't think of a
    # better way. Besides the point though.
    for rx, repl in _replacements:
        text = re.sub(rx, repl, text)

    # return truncated / padded version of string
    return text[:width] + ' ' * max(0, width - len(text))

class Obj():
    """I'm just an object with some attributes"""
    def __init__(self, **kw):
        self.__dict__.update(kw)

o = Obj(x="I am X, and I'm too long",
        y="ÄÖÜ Ich bin auch zu lang")
z = 'Pad me!'

format_spec = '<{o.x:6}>\n<{o.y:6}>\n<{z:10}>'

# Standard string formatting
print('Standard string formatting:')
print(format_spec.format(o=o, z=z))

# Demonstrate fix_format()
print('\nWanted output:')
print('<' + fix_format(o.x, 6) + '>')
print('<' + fix_format(o.y, 6) + '>')
print('<' + fix_format(z, 10) + '>')

##### This is where my struggle begins. #####

class MagicString(Formatter):
    '''Dummy implementation to have something that runs'''
    def __init__(self, format_spec):
        self.spec = format_spec
        super().__init__()

    def format(self, **kw):
        return(self.vformat(self.spec, [], kw))

    def format_field(self, v, s):
        if isinstance(v, str) and s.isdigit():
            return fix_format(v, int(s))

fmt = MagicString(format_spec)
print('\nReal output:')
print(fmt.format(o=o, z=z))

# Weirdly, somewhere on the way the standard formatting kicks in, too, as
# the 'Pad me!' string does get padded (which must be some postprocessing,
# as the string is still unpadded when passed into get_field())

