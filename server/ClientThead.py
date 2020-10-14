class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket, stream):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.clientAddress = clientAddress
        self.stream = stream
        self.lock = threading.Lock()
        self.frame = stream.get()

        print("New connection added: ", clientAddress)
        self.stopped = False

    def run(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        print("Connection from : ", self.clientAddress)

        while not self.stopped:

            self.lock.acquire()

            try:
                if not self.stream.empty():
                   
                    self.frame = self.stream.get()

                    a = pickle.dumps(self.frame, 0)
                    message = struct.pack("Q", len(a)) + a
                    self.csocket.sendall(message)
            except Exception as e:
                print(e)
            finally:
                logging.debug("Released a lock")

                self.lock.release()

        cam.release()
        print("Client at ", self.clientAddress, " disconnected...")
