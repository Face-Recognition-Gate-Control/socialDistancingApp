import struct
import socket
import time
import os
import json
import abc
from typing import List, Type, Callable, NewType
from io import BufferedReader
import mimetypes
import math
import uuid

from StreamHelpers import *


HOST = 'localhost'    # The remote host
PORT = 9876         # The same port as used by the server

# TOTAL PAYLOAD SIZE
PAYLOAD_LENGTH = 4

# SIZE OF THE PAYLOAD IDENTIFIER
PAYLOAD_NAME_LENGTH = 4

# SIZE OF SEGMENTS HEADER
SEGMENTS_LENGTH = 4

# SIZE OF JSON PAYLOAD
JSON_LENGTH = 4

# The order of the bytes from the stream little or big
BYTE_ORDER = "big"

# Encoding for strings
ENCODING = "UTF-8"

TMP_DIR = "./tmp/"

# Mimics Python socket.sendall signature
Send_function = NewType('Send_function', Callable[[bytes, int], None])
Recieve_function = NewType('Recieve_function', Callable[[bytes, int], None])

# RESPONSIBLE FOR CONNECTION TO SERVER


class FractalClient:

    def __init__(self, host: str, port: int, reader):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.reader = reader

    # Sends the payload to the server
    def send_payload(self, payload):
        payload.write_to_stream(self.socket.sendall)

    # Read a payload from the server connection stream
    def read_payload(self):
        return self.reader.read(self.socket.recv)


# RESPONSIBLE FOR READING DATA FROM SERVER
class FractalReader:

    # Reads the full payload from the stream of the provided byte stream read function.
    # It returns the payload when it is finished reading
    # THIS IS BLOCKING
    def read(self, reciever: Recieve_function):
        # Length of paylaod in bytes
        payload_length = read_int(reciever)

        # Read payload name length
        payload_name_length = read_int(reciever)
        # Read payload name
        payload_name = read_bytes_to_string(
            reciever, payload_name_length)

        # Read segments length
        segment_length = read_int(reciever)
        # Read segments
        segments = read_bytes_to_string(
            reciever, segment_length)

        # Read JSON payload length
        json_length = read_int(reciever)
        # Read JSON payload
        json_data = read_bytes_to_string(
            reciever, json_length)

        # parse JSON payload data
        payload_data = json.loads(json_data)

        # parse segment JSON payload data
        parsed_segments = json.loads(segments)

        return RecievedPayload(payload_data, self.readSegments(reciever, parsed_segments))

    # Reads the actual file segments parsed from header, and builds
    # the dictionary for it to retrieve the files with provided meta data.
    def readSegments(self, reciever, parsed_segments):
        segmentDict = {}
        for segment in parsed_segments:
            segment_key = next(iter(segment.keys()))
            segment_meta = next(iter(segment.values()))
            size = int(segment_meta["size"])
            file_name = ""
            if Segment._META_KEY_FILE_NAME in segment_meta:
                file_name = segment_meta[Segment._META_KEY_FILE_NAME]
            else:
                file_name = str(uuid.uuid1())

            file_full_path = TMP_DIR + file_name
            tempt_segment_file = open(TMP_DIR + file_name, 'wb')
            buffer = 4096
            remaining = size
            received_bytes = reciever(min(buffer, remaining))
            while (received_bytes):
                remaining -= len(received_bytes)
                tempt_segment_file.write(received_bytes)
                received_bytes = reciever(min(buffer, remaining))
            tempt_segment_file.close()
            segmentDict[segment_key] = RecivedSegment(
                segment_meta, file_full_path)
        return segmentDict

# Holder class for received segments


class RecivedSegment:
    def __init__(self, segment_meta, file_name):
        self.segment_meta = segment_meta
        self.file_name = file_name

# Holder class for received payloads, includes payload data< JSON and
# segments< Files


class RecievedPayload:
    def __init__(self, payload_data, segments):
        self.payload_data = payload_data
        self.segments = segments


class Segment(metaclass=abc.ABCMeta):

    _META_KEY_SIZE = "size"
    _META_KEY_MIME_TYPE = "mime_type"
    _META_KEY_FILE_NAME = "file_name"

    def __init__(self):
        super().__init__()
        self.segment_meta = {}

    @ abc.abstractmethod
    def write_to_stream(self, send_function: Send_function):
        pass

    def get_segment_size(self):
        return self.segment_meta.get(self._META_KEY_SIZE)

    def get_meta(self):
        return self.segment_meta

    def set_segment_size(self, size: int):
        self.segment_meta[self._META_KEY_SIZE] = size

    def set_segment_mime_type(self, mimeType: str):
        self.segment_meta[self._META_KEY_MIME_TYPE] = mimeType

    def set_segment_file_name(self, file_name: str):
        self.segment_meta[self._META_KEY_FILE_NAME] = file_name


class JsonSegment(Segment):

    def __init__(self, json_string: str):
        super().__init__()
        self.json_string = json_string
        self.set_segment_size(len(json_string))
        self.set_segment_mime_type("application/json")

    def write_to_stream(self, send_function: Send_function):
        send_function(bytes(self.json_string, ENCODING))


class FileSegment(Segment):

    def __init__(self, file: BufferedReader):
        super().__init__()
        self._file = file
        self.set_segment_size(os.path.getsize(file.name))
        self.set_segment_mime_type(mimetypes.guess_type(file.name)[0])
        file_name = file.name.split("/")
        self.set_segment_file_name(file_name[-1])

    def write_to_stream(self, send_function: Send_function):
        reader = self._file.read(256)

        while (reader):
            send_function(reader)
            reader = self._file.read(256)

        self._file.close()


class Payload:

    def __init__(self, payload_name: str):
        self.payload_name = payload_name
        self.segments = {}
        self.json_body = "{}"

    # Adds a segment to the payload
    def add_segment(self, name: str, segment: Segment):
        self.segments[name] = segment

    # Adds json string data payload: string from json.dumps
    def add_json_data(self, json_string):
        self.json_body = json_string

    # Writes the payload to the stream of the socket
    def write_to_stream(self, send_all: Send_function):
        # HOLDS TOTAL PAYLOAD SIZE
        total_payload_size = 0

        # PAYLOAD NAME
        payload_name_length = len(self.payload_name)
        payload_name_bytes_size = payload_name_length.to_bytes(
            PAYLOAD_NAME_LENGTH, BYTE_ORDER)
        payload_name_bytes = bytes(self.payload_name, ENCODING)

        total_payload_size += payload_name_length

        # SEGMENTS SIZE AND META
        segments = []
        segment_size = 0
        for (segment_identifier, segment) in self.segments.items():
            segments.append({segment_identifier: segment.get_meta()})
            segment_size += segment.get_segment_size()
            total_payload_size += segment.get_segment_size()
        # print(f"TOTAL SEGMENT SIZE ", segment_size)
        # PARSE META TO JSON AND GET LENGTH
        segment_json = json.dumps(segments)
        segment_meta_length = len(segment_json)
        segment_meta_byte_size = segment_meta_length.to_bytes(
            SEGMENTS_LENGTH, BYTE_ORDER)
        segment_meta_bytes = bytes(segment_json, ENCODING)

        total_payload_size += segment_meta_length

        # JSON BODY
        json_body_length = len(self.json_body)
        json_body_byte_size = json_body_length.to_bytes(
            JSON_LENGTH, BYTE_ORDER)
        json_body_bytes = bytes(self.json_body, ENCODING)
        total_payload_size += json_body_length

        total_payload_bytes = total_payload_size.to_bytes(
            PAYLOAD_LENGTH, BYTE_ORDER)

        ## START WRITING ##

        # TOTAL PAYLOAD SIZE
        send_all(total_payload_bytes)

        # PAYLOAD NAME SIZE
        send_all(payload_name_bytes_size)
        # PAYLOAD NAME
        send_all(payload_name_bytes)

        # SEGMENT META SIZE
        send_all(segment_meta_byte_size)
        # SEGMENT META
        send_all(segment_meta_bytes)

        # JSON f
        send_all(json_body_byte_size)
        # JSON DATA
        send_all(json_body_bytes)

        # SEGMENTS
        for segment in self.segments.values():
            segment.write_to_stream(send_all)


# CONNECT TO SERVER
client = FractalClient(HOST, PORT, FractalReader())

# CREATE PAYLOADS
payload = Payload("authentication")
payload.add_json_data(json.dumps({"identificationId": "RANDOMSTRING HERE"}))

payload.add_segment("JSONSEGMENT", JsonSegment(json.dumps({
    "arraydata": [
        {
            "data": "text",
            "number": 20000
        }
    ], "field": "some field data"})))

payload.add_segment("FILESEGMENT", FileSegment(
    open("./files/pom.xml", "rb")))

# SEND PAYLOAD
client.send_payload(payload)

# RECEIVE PAYLOADS
receivedPyloadisHere = client.read_payload()
