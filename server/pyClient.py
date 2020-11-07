import threading
import socket
import time
import pickle


class ClientPy(threading.Thread):
    def __init__(self, HOST, PORT, sendQueue):
        threading.Thread.__init__(self)
        self.HOST = HOST
        self.PORT = PORT
        self.queue = sendQueue

    def run(self):
        connected = False

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while not connected:
            try:
                s.connect((self.HOST, self.PORT))

            except ConnectionRefusedError as e:
                connected = False

            else:
                print("connected")
                connected = True

        while True:

            try:

                if not self.queue.empty():
                    data = self.queue.get()
                    data = pickle.dumps(data)
                    s.sendall(pickle.dumps(1))
                    s.sendall(data)
                    time.sleep(0.1)

            except Exception as e:
                print(e)

        s.close()
