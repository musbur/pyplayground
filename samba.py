from smb.SMBConnection import SMBConnection

conn = SMBConnection('manuser', 'balzers',
              'deham01in015',
              '',
              use_ntlm_v2=True,
              sign_options=SMBConnection.SIGN_WHEN_SUPPORTED,
              is_direct_tcp=True)
connected = conn.connect('192.168.232.88', 139)
