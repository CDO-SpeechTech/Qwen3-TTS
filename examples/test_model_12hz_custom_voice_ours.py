# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- Single (with instruct) --------
    torch.cuda.synchronize()
    t0 = time.time()

    # syn_text_single = "여러분의 시간이 소중한 만큼, 의미있게 사용되었으면 좋겠어요."
    # syn_text_single = "죄송합니다. 이 상품은 신규 가입이 중단되었습니다. '참 쉬운 가족 결합'을 안내해 드릴까요?"
    syn_text_single = "오늘 정말 즐거웠어요. 이렇게 좋은 시간 보낼 수 있어서 기뻐요."
    # syn_text_single = "왜 아무도 내 생각에 동조하지 않는 거지? 이 더러운 자본주의 돼지놈들!"
    # syn_text_single = "이용중인 PPL대출 DB지수연계증권투자신탁SEK-46호[ELS-파생형]A,연장 신청 대출 금액은, 200000000원,기준금리는, CD연동대출기준금리.가산금리는, 1.7%가 적용되어금일 기준으로 합산한 총 금리는, 4.5 4% 입니다."
    # syn_text_single = "따뜻한 봄바람이 불어오는 어느 늦은 오후에 나는 조용한 동네의 작은 책방 안 창가에 앉아 나무로 된 책상 위에 펼쳐진 오래된 시집을 바라보며 그 안에 담긴 시인의 감성과 시간의 무게를 천천히 음미하고 있었고 창밖으로는 아이들이 골목길을 뛰어다니며 웃고 장난치는 소리가 들려왔으며 바람을 타고 퍼지는 꽃내음이 내 코끝을 간질이고 있었으며 책방 한쪽에서는 주인아저씨가 라디오에서 흘러나오는 클래식 음악에 맞춰 리듬을 타듯 조용히 책 정리를 하고 있었고 나는 이 모든 풍경이 마치 한 편의 시처럼 조화롭게 느껴졌으며 그 순간 나는 일상의 소란과 복잡함을 잠시 내려놓고 이 조용하고 따뜻한 공간 속에서 내가 진짜로 원하는 삶이 무엇인지 내가 앞으로 어떤 모습으로 살아가고 싶은지에 대한 생각들을 조용히 떠올려보며 글을 쓰고 싶다는 오랜 꿈을 다시 떠올렸고 비록 지금은 바쁜 일상과 현실적인 문제들로 인해 그 꿈을 잊고 살아가고 있었지만 이런 고요하고 감성적인 시간 속에서는 내가 누구인지 내가 무엇을 사랑하는지 다시금 명확하게 느낄 수 있었고 그것만으로도 충분히 큰 위로가 되었으며 삶이라는 것은 거창한 성공이나 눈에 띄는 성취만으로 채워지는 것이 아니라 이런 사소한 순간들 속에서 느껴지는 작지만 확실한 감정들과 마음의 떨림들 속에서 진짜 의미를 찾을 수 있다는 사실을 다시 한번 마음 깊이 새기게 되었으며 그렇게 나는 오랜만에 여유롭고도 뿌듯한 기분으로 그 책방을 나서며 오늘 하루도 꽤 괜찮았다고 스스로에게 조용히 말하고 있었다."
    wavs, sr = tts.generate_custom_voice(
        text=syn_text_single,
        language="Korean",
        speaker="Sohee",
        instruct="매우 기쁜 어조로 말하다.",
        # non_streaming_mode=False
    )
    # wavs, sr = tts.generate_custom_voice(
    #     text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    #     language="Chinese",
    #     speaker="Vivian",
    #     instruct="用特别愤怒的语气说",
    # )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # # -------- Batch (some empty instruct) --------
    # texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", "She said she would be here by noon."]
    # languages = ["Chinese", "English"]
    # speakers = ["Vivian", "Ryan"]
    # instructs = ["", "Very happy."]

    # torch.cuda.synchronize()
    # t0 = time.time()

    # wavs, sr = tts.generate_custom_voice(
    #     text=texts,
    #     language=languages,
    #     speaker=speakers,
    #     instruct=instructs,
    #     max_new_tokens=2048,
    # )

    # torch.cuda.synchronize()
    # t1 = time.time()
    # print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    # for i, w in enumerate(wavs):
    #     sf.write(f"qwen3_tts_test_custom_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()
