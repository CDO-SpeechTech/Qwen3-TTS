import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
# MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/"
MODEL_PATH = "sft_output/Qwen3-TTS-12Hz-1.7B-Base/exp1/checkpoint-epoch-7"
# MODEL_PATH = "sft_output/Qwen3-TTS-12Hz-0.6B-Base/exp1/checkpoint-epoch-7"
tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_3",
)

# 안내체 목소리
# "m_baro", "m_woojoo", "m_minwoo", "m_doyun", "m_taeho"
# "f_sua", "f_bomi", "f_inna", "f_hayun", "f_nuri"
# "f_miso" (B2C 라이선스만 있음)
# "f_lili" (여자아이)
# 
# 대화체 목소리
# "f_sunah", "f_haeun", "f_hyunju", "f_jiwon"
# "m_junho", "m_kangwoo" "m_seongmin",
# "f_junga", "m_hansu": (노인)
# "m_nuri" (남자아이)

# LLM상담봇
# "30_f_sua"

print("tts model loaded")
torch.cuda.synchronize()
t0 = time.time()
wavs, sr = tts.generate_custom_voice(
    # text="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    # text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    # text="明日はきっといい日になるよ。",
    # text= "Italiano è una lingua bellissima che suona come una melodia.",
    # text="A felicidade está nas pequenas coisas do dia a dia.",
    # text="왜 아무도 내 생각에 동조하지 않는 거지? 이 더러운 자본주의 돼지놈들!",
    # text="미안해요. TV 보여주고 그 시간에 쇼츠 보느라 시간이 다 가는 게 한심하고, 아이에게도 미안해요.",
    # text="오늘 정말 즐거웠어요. 이렇게 좋은 시간 보낼 수 있어서 기뻐요.",
    # text="성과공유회 처음으로 비투비 성과에 대해서도 알려드리겠습니다!",
    text="안녕하세요~ LG유플러스 임직원 여러분, 2026년 1분기 성과공유회를 시작 하겠습니다!!",# 오늘의 순서는 25년 경영실적을 공유 드리고, 컨슈머, 엔터프라이즈 부문 및 기본기 조직들의 중점 추진과제를 소개해드리려고 합니다! 그럼 먼저, 26년 1분기 경영성과를 같이 볼까요? 지난해 모든 임직원 분들의 열정과 노력 아래, 매출 11조 4726억원으로 전년 대비 4.0%, 영업이익은 9563억원으로 +3.7% 증가했습니다. 마켓쉐어는 26.0%로 전년 대비 0.81% 증가하면서 3사 중 가장 높은 상승률을 보였고, 매출 성장률 또한 KT와 격차를 벌어졌습니다! 이익률을 확보하기 어려웠던 작년에도 저희 유플러스는 전년대비 0.1%만 감소해서, SK와 KT보다 적은 감소폭을 유지했습니다!! MNO 핸드셋 가입자는 유플러스는 29만2천, KT는 3만5천 순증한 반면, SK는 9만2천 순감하였습니다. 이에 당사는 누적가입자 기준 마켓쉐어 24.0%로 전년대비 +0.8%p 증가하였습니다. 가입자는 계획대비 초과 순증하였습니다. MVNO 핸드셋 가입자는 4만8천 순증으로 유플러스가 가장 큰 성장을 유지하고 있습니다. 인터넷 순증 마켓쉐어도 성장 추세를 유지하며, 순증 1위를 지속하고 있습니다. 성과공유회 처음으로 B2B 성과에 대해서도 알려드리겠습니다! 마켓쉐어는 24년 조금 하락했지만, ‘25년 반등하였고, 매출은 점진적으로 증가하여, 2조3802억을 달성하였습니다! B2B 성과 중에서도, IDC성과가 두드러지게 나타나고 있습니다! 마켓쉐어는 1분기 19.2%에서 4분기 25.3%로 성장하며, 매출은 약 55% 성장하며 4분기 1353억을 달성하였습니다! 모두 박수한번 쳐주세요! 25년, 전 임직원의 열정과 노력으로 좋은 성과를 만들어 냈습니다. 하지만 경쟁사의 보안이슈가 다시 한번 불거지며 기본기에 대한 중요성은 더욱 대두되었습니다. 구성원 모두 다시 한번 열정 넘치는 노력, 그리고 기본기를 최우선으로하여 26년에도 성과, 기본기 모두 잡을 수 있는 한 해가 되시길 기대하겠습니다. 다음 순서로, Consumer, Enterprise 부문 및 기본기 조직들의 26년 중점 추진과제를 소개해드리겠습니다.",
    language="Korean",
    speaker="f_sunah",
    # instruct="非常快速。",
    # instruct="Very happy and expressive.",
    # instruct="Very sad and gloomy.",
    non_streaming_mode=True
)
torch.cuda.synchronize()
t1 = time.time()
print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

sf.write("output.wav", wavs[0], sr)
