from email import message_from_binary_file
from email.message import EmailMessage
from email.utils import make_msgid
from email.parser import BytesParser
import smtplib

def create_message():
    '''Return an EMailMessage with HTML content with image displayed
    inline (in Outlook)'''
    msg = EmailMessage()
    msg['From'] = 'daniel.haude@nexperia.com'
    msg['To'] = 'daniel.haude@nexperia.com'
    msg['Subject'] = 'Ausschussbericht'

    submsg = EmailMessage()

    imgid = make_msgid('img1')
    submsg.add_alternative('''
    <html>
     <body>
      Text above img<br>
      <img src="cid:{imgid}"><br>
      Text below img
     </body>
    </html>'''.format(imgid=imgid[1:-1]), subtype='html')

    with open('logo.png', 'rb') as fh:
        buf = fh.read()

    msg.add_related(submsg)
    msg.add_related(buf, maintype='image', subtype='png', cid=imgid)
    return msg

def iter_parts(msg, depth=0):
    '''Recursively list contents of multi-part email'''
    print('   ' * depth, msg.get_content_type())
    if msg.is_multipart():
        for p in msg.iter_parts():
            iter_parts(p, depth+1)

if __name__ == '__main__':
    with open('from_outlook.txt', 'rb') as fh:
        msg1 = message_from_binary_file(fh, EmailMessage)
    print('From Outlook')
    iter_parts(msg1)

    msg2 = create_message()
    print('Homemade')
    iter_parts(msg2)

def process_message(msg):
    SMTPRELAY = 'smtprelay.de-hbg01.nexperia.com'

    if True:
        s = smtplib.SMTP(SMTPRELAY)
        s.send_message(msg)
    else:
        print(msg.as_string())

