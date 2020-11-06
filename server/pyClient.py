from threading import Thread
import socket
import time
import pickle
class ClientPy(Thread):

    def __init__(self,HOST,PORT,sendQueue):
        Thread.__init__(self)
        self.HOST = HOST
        self.PORT = PORT
        self.queue = sendQueue
    


    def run(self):
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST,PORT))

        while True:
    
	        data = self.queue.get()
            data = pickle.dumps(data)
            s.sendall(pickle.dumps(1))
	        s.sendall(data)
            time.sleep(0.1)
        s.close()
        

