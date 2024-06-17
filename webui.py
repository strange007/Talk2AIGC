import gradio as gr
from pathlib import Path
import time
import os
import soundfile as sf
from pydub import AudioSegment
import whisper
from transformers import AutoModel, AutoTokenizer
from paddlespeech.cli.tts.infer import TTSExecutor
import librosa

#语音输入
inputs_path = "/mnt/workspace/inputs"
#语音输出
outputs_path = "/mnt/workspace/outputs"
#文字转语音
tts = TTSExecutor()

tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/workspace/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()
model = model.eval()

#对话记录
history_list = []


# 语音转换
def convert_wav(input_file, output_file):
    sound = AudioSegment.from_file(input_file)
    # set frame rate to 16000
    sound = sound.set_frame_rate(16000)
    # set channels to mono (1)
    sound = sound.set_channels(1)
    # set sample width to 2 (16 bit)
    sound = sound.set_sample_width(2)
    sound.export(output_file, format="wav")



def chat(input_audio):
    
    # 麦克风输入为空
    if input_audio is None:
        response = "你有什么想问的吗？"
        txt=""

    # 麦克风输入不为空，使用语音输入数据进行对话
    if input_audio is not None:
        rate, audio = input_audio
        is_exists = os.path.exists(inputs_path)
        if not is_exists:
            os.makedirs(inputs_path)
        # 将输入的语音保存为本地的wav文件
        input_audio_file_name = inputs_path + "/input_audio" + str(time.time()) + ".wav"
        sf.write(input_audio_file_name, audio, rate)
        # 将输入wav文件转换文件的采样率为16000，单声道，采样深度为16位的wav文件
        convert_audio_file_name = inputs_path + "/convert_audio" + str(time.time()) + ".wav"
        convert_wav(input_audio_file_name,convert_audio_file_name)

        # 用whisper进行语音转换为文字
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(convert_audio_file_name, fp16=False, language="Chinese")
        txt=result["text"]

    # 文本输入不为空，使用文本输入数据进行对话
    if txt != "":
        response, history = model.chat(tokenizer, txt, history=history_list)
        history_list.append(history[len(history) - 1])
        
    # 创建输出文件夹
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    # 语音输出文件
    audio_file_name = outputs_path + "/results" + str(hash(txt)) + str(time.time()) + ".wav"
    # 转换回答为语音
    tts(text=response, am="fastspeech2_mix", lang="mix", output=audio_file_name)

    # 拼接历史对话
    if len(history_list) != 0:
        separator = " "
        combined_history = separator.join([f"User：{x[0]} \n\n Answear:{x[1]} \n\n" for x in history_list])
    else:
        combined_history = ""

    audio, sr = librosa.load(path=audio_file_name)

    return (sr, audio),response, combined_history


input_text = gr.Microphone(label="请说出你要问的问题")
output_multimodal = [gr.Audio(label="语音输出"), gr.Textbox(label="回答", max_lines=30), gr.Text(label="历史记录")]

interface = gr.Interface(
    fn=chat,
    inputs=input_text,
    outputs=output_multimodal,
    title="语音对话系统",
)

interface.launch(share=True)