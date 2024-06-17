import gradio as gr
from pathlib import Path
import time
import wave
import os
import soundfile as sf
import io
from pydub import AudioSegment
import torch
import whisper
from threading import Thread
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from paddlespeech.cli.tts.infer import TTSExecutor
import librosa
import shutil
import numpy as np

#语音输入
inputs_path = "/mnt/workspace/inputs"
#语音输出
outputs_path = "/mnt/workspace/outputs"
temp_path = "/mnt/workspace/outputs/temp"

#文字转语音
tts = TTSExecutor()
#语音转文字
whisper_model = whisper.load_model("base")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/workspace/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda().to(device)

#对话记录
history_list = ""
round=0

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


# 合并两个语音，输出合并后的文件路径
def combined_audio(intput_path1,intput_path2):
    infiles=[intput_path1,intput_path2]
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
        
    output = wave.open(intput_path1, 'wb')
    output.setparams(data[0][0])
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    output.close()
    audio, sr = librosa.load(path=intput_path1)
    return (sr, audio),intput_path1

def get_audio_length(file_path):
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = frames / float(rate)
        return int(wav_length+1)


def get_bytes(path):

    file = open(path, 'br')

    # Convert the binary response content to a byte stream
    byte_stream = io.BytesIO(file.read())

    # Read the audio data from the byte stream
    audio = AudioSegment.from_file(byte_stream, format="wav")

    # Export the audio as WAV format
    sample_width = audio.sample_width
    sample_rate = audio.frame_rate
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.int16)
    
    file.close()

    return sample_rate, audio_data



def chat(input_audio):
    global history_list

    # 麦克风输入为空
    if input_audio is None:
        response = "你有什么想问的吗？"
        # 语音输出文件
        audio_file_name = outputs_path + "/results" + str(time.time()) + ".wav"
        tts(text=response, am="fastspeech2_mix", lang="mix", output=audio_file_name)
        txt=""
        audio, sr = librosa.load(path=audio_file_name)

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
        result = whisper_model.transcribe(convert_audio_file_name, fp16=False, language="Chinese")
        txt=result["text"]

    # 文本输入不为空，使用文本输入数据进行对话
    if txt != "":
        print(txt)
        
        history_list += "\nUser："+txt
        input_ids = tokenizer(txt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = dict(input_ids, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        turn_count = 0
        generated_text = ""
        last_output=""
        for new_text in streamer:
            if turn_count == 0:
                turn_count += 1
                history_list += "\nanswear:"
                continue
            elif new_text=="":
                continue
            history_list += new_text
            generated_text += new_text
            response = generated_text[1:]

            # 语音输出文件
            namestr=str(hash(new_text)) + str(time.time())
            audio_file_name = temp_path + "/results" + namestr + ".wav"
            
            try:
                # 转换回答为语音
                tts(text=new_text, am="fastspeech2_mix", lang="mix", output=audio_file_name)
                
                if turn_count==1:
                    turn_count+=1
                    shutil.move(audio_file_name, outputs_path)
                    last_output= outputs_path + "/results" + namestr + ".wav"
                    audio, sr = librosa.load(path=audio_file_name)
                    audio_data = (sr, audio)
                    
                else:
                    audio_data, last_output=combined_audio(last_output,audio_file_name)
                    audio, sr = librosa.load(path=last_output)
                    

            except Exception as e:
                # 有时候tts时文本是某些特殊字符时可以跳过
                print('\n产生错误',e,"\n")
                continue
                
            # sample_rate, audio_data=get_bytes(last_output)

            yield (sr, audio),response,history_list
        

    # 文本输入为空
    if txt == "":
        # audio=get_bytes(audio_file_name)
        yield (sr, audio), response, history_list
        

input_text = gr.Microphone(label="请说出你要问的问题")
output_multimodal = [gr.Audio(label="语音输出"), gr.Textbox(label="回答", max_lines=30), gr.Text(label="历史记录")]

# 创建一个音频录制按钮和状态文本
interface = gr.Interface(
    fn=chat,
    inputs=input_text,
    outputs=output_multimodal,
    title="语音对话系统",
    live=True
)

# 预加载语音模型
try :
    result = whisper_model.transcribe("/mnt/workspace/inputs/convert_audio1712632464.2535655.wav", fp16=False, language="Chinese")
    tts(text="", am="fastspeech2_mix", lang="mix", output="/mnt/workspace/outputs/temp/temp.wav")

finally:

    # 设置页面标题并运行
    interface.queue().launch(share=True)