from threading import Thread
import socket
class ClientPy(Thread):
    def __init__(self,HOST,PORT,sendQueue):
        Thread.__init__(self)
        self.HOST = HOST
        self.PORT = PORT
    


    def run():
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST,PORT))

        while True:
	        data = sendQueue.get()
	        s.send(data)
	        reply = s.recv(1024)
		    if reply == 'Terminate':
			    break
		    print reply
