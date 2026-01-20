      
from agent.agent import LLM,VisionLLM
from voice.voice import VoiceInput,VoiceOutput
# import signal
import sys
# import os

TASK_PRIORITY = {
    'start_record': -2,
    'end_record': -2,
    'start_execution': -5,
    'stop_execution': -5,
    'normal': -5,
    'pause': -3,
    'exit': -5,
    'wait': -4,
    "start_processing": -2,
    "start_processing": -2,
    "get_from_camera":-2
}

if __name__ == "__main__":
    voice_input = VoiceInput()
    voice_output = VoiceOutput()
    # llm = LLM()
    llm = VisionLLM()
    llm.set_comands(TASK_PRIORITY)

    @llm.llm_wrapper
    def handle_message(msg, token):
        if token == "get_from_camera":
            caption = llm.invoke(msg)
            voice_output.say(caption,-6)
        voice_output.say(msg, TASK_PRIORITY[token])

    def close_all(*args,**kwargs):
        voice_input.close()
        voice_output.close()
        llm.close()

    voice_input.add_message_callback(handle_message)
    voice_input.add_on_exit_callback(close_all)

    try:
        voice_input.start()
        llm.start()
        voice_output.process_queue()
    except KeyboardInterrupt:
        close_all()
        sys.exit(0)
