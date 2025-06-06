import base64
import struct
import numpy as np
import traceback

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading


class VoiceModel(BaseModel):
    timestamp: int
    buffer: str


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])

        self.tlock = threading.Lock()

    def test(self, voice: VoiceModel):
        try:
            timestamp = voice.timestamp
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            # if wav == 0:
            #     samplerate, data = read("dummy.wav")
            #     unpackedData = data
            # else:
            #     unpackedData = np.array(
            #         struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)
            #     )

            unpackedData = np.array(struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)).astype(np.int16)
            # print(f"[REST] unpackedDataType {unpackedData.dtype}")

            self.tlock.acquire()
            changedVoice = self.voiceChangerManager.changeVoice(unpackedData)
            #self.tlock.release()
            if self.tlock.locked():
                self.tlock.release()

            #changedVoiceBase64 = base64.b64encode(changedVoice[0]).decode("utf-8")
            # if changedVoice[0] is not None:
            #     changedVoiceBase64 = base64.b64encode(changedVoice[0]).decode("utf-8")
            # else:
            #     changedVoiceBase64 = ""

            # TA change: This might still be producing audio! can interfere with other app!
            if changedVoice[0] is not None and len(changedVoice[0]) > 0:
                changedVoiceBase64 = base64.b64encode(changedVoice[0]).decode("utf-8")
            else:
                silent_audio = np.zeros(160, dtype=np.int16)  # ~10ms at 16kHz
                changedVoiceBase64 = base64.b64encode(silent_audio.tobytes()).decode("utf-8")

            data = {"timestamp": timestamp, "changedVoiceBase64": changedVoiceBase64}

            json_compatible_item_data = jsonable_encoder(data)
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            self.tlock.release()
            return str(e)
