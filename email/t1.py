import smtplib
from email import encoders
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage

strFrom = 'zzzzzz@gmail.com'
strTo = 'xxxxx@gmail.com'

# Create the root message 

msg = MIMEMultipart('related')
msg['From'] = 'daniel.haude@nexperia.com'
msg['To'] = 'daniel.haude@nexperia.com'
msg['Subject'] = 'Ausschussbericht'
msg.preamble = 'Multi-part message in MIME format.'

msgAlternative = MIMEMultipart('alternative')
msg.attach(msgAlternative)

msgText = MIMEText('Alternative plain text message.')
msgAlternative.attach(msgText)

msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>KPI-DATA!', 'html')
msgAlternative.attach(msgText)

#Attach Image 
fp = open('logo.png', 'rb') #Read image 
msgImage = MIMEImage(fp.read())
fp.close()

# Define the image's ID as referenced above
msgImage.add_header('Content-ID', '<image1>')
msg.attach(msgImage)


def process_message(msg):
    SMTPRELAY = 'smtprelay.de-hbg01.nexperia.com'

    if True:
        s = smtplib.SMTP(SMTPRELAY)
        s.send_message(msg)
    else:
        print(msg.as_string())

process_message(msg)
