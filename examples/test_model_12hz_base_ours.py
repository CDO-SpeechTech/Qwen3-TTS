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
import os
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base/"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
    ensure_dir(OUT_DIR)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Reference audio(s)
    # ref_audio_path_1 = "pb_1.wav"
    ref_audio_path_1 = "samples_for_zstts/F_BOMI_1_1000.wav"
    # ref_audio_path_2 = "samples_for_zstts/M_WOOJOO_1_1000.wav"

    ref_audio_single = ref_audio_path_1
    # ref_audio_batch = [ref_audio_path_1, ref_audio_path_2]

    ref_text_single = "구매일로부터 7일 이내에 구독 철회 및 환불이 가능합니다."
    # ref_text_single = "네, 우리 김병준 매니저 바로 확인해서 연락드릴 수 있도록 다시 한번 전달하겠습니다."
    # ref_text_batch = [
    #     "구매일로부터 7일 이내에 구독 철회 및 환불이 가능합니다.",
    #     "구매일로부터 7일 이내에 구독 철회 및 환불이 가능합니다.",
    # ]

    # Synthesis targets
    # syn_text_single = "안녕하세요~ LG유플러스 임직원 여러분, 2026년 1분기 성과공유회를 시작 하겠습니다!!"# 오늘의 순서는 25년 경영실적을 공유 드리고, 컨슈머, 엔터프라이즈 부문 및 기본기 조직들의 중점 추진과제를 소개해드리려고 합니다! 그럼 먼저, 26년 1분기 경영성과를 같이 볼까요? 지난해 모든 임직원 분들의 열정과 노력 아래, 매출 11조 4726억원으로 전년 대비 4.0%, 영업이익은 9563억원으로 +3.7% 증가했습니다. 마켓쉐어는 26.0%로 전년 대비 0.81% 증가하면서 3사 중 가장 높은 상승률을 보였고, 매출 성장률 또한 KT와 격차를 벌어졌습니다! 이익률을 확보하기 어려웠던 작년에도 저희 유플러스는 전년대비 0.1%만 감소해서, SK와 KT보다 적은 감소폭을 유지했습니다!! MNO 핸드셋 가입자는 유플러스는 29만2천, KT는 3만5천 순증한 반면, SK는 9만2천 순감하였습니다. 이에 당사는 누적가입자 기준 마켓쉐어 24.0%로 전년대비 +0.8%p 증가하였습니다. 가입자는 계획대비 초과 순증하였습니다. MVNO 핸드셋 가입자는 4만8천 순증으로 유플러스가 가장 큰 성장을 유지하고 있습니다. 인터넷 순증 마켓쉐어도 성장 추세를 유지하며, 순증 1위를 지속하고 있습니다. 성과공유회 처음으로 B2B 성과에 대해서도 알려드리겠습니다! 마켓쉐어는 24년 조금 하락했지만, ‘25년 반등하였고, 매출은 점진적으로 증가하여, 2조3802억을 달성하였습니다! B2B 성과 중에서도, IDC성과가 두드러지게 나타나고 있습니다! 마켓쉐어는 1분기 19.2%에서 4분기 25.3%로 성장하며, 매출은 약 55% 성장하며 4분기 1353억을 달성하였습니다! 모두 박수한번 쳐주세요! 25년, 전 임직원의 열정과 노력으로 좋은 성과를 만들어 냈습니다. 하지만 경쟁사의 보안이슈가 다시 한번 불거지며 기본기에 대한 중요성은 더욱 대두되었습니다. 구성원 모두 다시 한번 열정 넘치는 노력, 그리고 기본기를 최우선으로하여 26년에도 성과, 기본기 모두 잡을 수 있는 한 해가 되시길 기대하겠습니다. 다음 순서로, Consumer, Enterprise 부문 및 기본기 조직들의 26년 중점 추진과제를 소개해드리겠습니다."
    syn_lang_single = "Korean"

    # syn_text_single = "여러분의 시간이 소중한 만큼, 의미있게 사용되었으면 좋겠어요."
    # syn_text_single = "죄송합니다. 이 상품은 신규 가입이 중단되었습니다. '참 쉬운 가족 결합'을 안내해 드릴까요?"
    # syn_text_single = "오늘 정말 즐거웠어요. 이렇게 좋은 시간 보낼 수 있어서 기뻐요."
    # syn_text_single = "왜 아무도 내 생각에 동조하지 않는 거지? 이 더러운 자본주의 돼지놈들!"
    syn_text_single = "성과공유회 처음으로 비투비 성과에 대해서도 알려드리겠습니다!",

    syn_text_batch = [
        # "안녕하세요~ LG유플러스 임직원 여러분, 2026년 1분기 성과공유회를 시작 하겠습니다!! . . . . .",
        # "오늘의 순서는 25년 경영실적을 공유 드리고, 컨슈머, 엔터프라이즈 부문 및 기본기 조직들의 중점 추진과제를 소개해드리려고 합니다! 그럼 먼저, 26년 1분기 경영성과를 같이 볼까요? . . . . .",
        # "지난해 모든 임직원 분들의 열정과 노력 아래, 매출 11조 4726억원으로 전년 대비 4.0%, 영업이익은 9563억원으로 +3.7% 증가했습니다. . . . . .",
        # "마켓쉐어는 26.0%로 전년 대비 0.81% 증가하면서 3사 중 가장 높은 상승률을 보였고, 매출 성장률 또한 KT와 격차를 벌려왔습니다! 그리고, 이익률을 확보하기 어려웠던 작년에도 유플러스만 유일하게 전년대비 1.0% 증가했습니다. . . . . .",
        # "MNO 핸드셋 가입자는 유플러스는 130만7천, KT는 178만8천 순증한 반면, SK는 97만9천 순감하였습니다. LG유플러스는 누적가입자 기준 마켓쉐어 24.1%로 전년대비 +1.0% 증가하였습니다. 가입자는 계획대비 초과 순증하였습니다. . . . . .",
        # "MVNO 핸드셋 가입자는 41만7천 순증으로 유플러스가 가장 큰 성장을 유지하고 있습니다.",
        # "인터넷 순증 마켓쉐어도 성장 추세를 유지하며, 순증 1위를 지속하고 있습니다.",
        # "성과공유회 처음으로 B2B 성과에 대해서도 알려드리겠습니다! 마켓쉐어는 24년 조금 하락했지만, 25년 반등에 성공하였고, 매출은 소폭 상승하여, 2조3802억원을 달성했습니다!",
        # "B2B 성과 중에서도, IDC성과가 두드러지게 나타나고 있는데, 마켓쉐어는 1분기 19.2%에서 4분기 25.3%로 성장하며, 매출은 약 55% 성장하며 4분기 1353억을 달성했습니다!",
        # "모두 박수한번 쳐주세요!",
        # "25년, 전 임직원의 열정과 노력으로 좋은 성과를 만들어 냈습니다. 하지만 경쟁사의 보안이슈가 다시 한번 불거지며 기본기에 대한 중요성은 더욱 대두되었습니다. 구성원 모두 다시 한번 열정 넘치는 노력, 그리고 기본기를 최우선으로 하여 26년에도 성과, 기본기 모두 잡을 수 있는 한 해가 되시길 기대하겠습니다.",
        # "다음 순서로, 컨슈머, 엔터프라이즈 부문 및 기본기 조직인 정보보안, 품질혁신, 안전환경 조직들의 26년 중점 추진과제를 소개해드리겠습니다.",

        "안녕하세요~ LG유플러스 임직원 여러분, 2026년 1분기 성과공유회를 시작 하겠습니다!!",
        "오늘의 순서는 25년 경영실적을 공유 드리고, 컨슈머, 엔터프라이즈 부문 및 기본기 조직들의 중점 추진과제를 소개해드리려고 합니다! 그럼 먼저, 26년 1분기 경영성과를 같이 볼까요?",
        "지난해 모든 임직원 분들의 열정과 노력 아래, 매출 11조 4726억원으로 전년 대비 4.0%, 영업이익은 9563억원으로 +3.7% 증가했습니다.",
        "마켓쉐어는 26.0%로 전년 대비 0.81% 증가하면서 3사 중 가장 높은 상승률을 보였고, 매출 성장률 또한 KT와 격차를 벌려왔습니다! 그리고, 이익률을 확보하기 어려웠던 작년에도 유플러스만 유일하게 전년대비 1.0% 증가했습니다.",
        "MNO 핸드셋 가입자는 유플러스는 130만7천, KT는 178만8천 순증한 반면, SK는 97만9천 순감하였습니다. LG유플러스는 누적가입자 기준 마켓쉐어 24.1%로 전년대비 +1.0% 증가하였습니다. 가입자는 계획대비 초과 순증하였습니다.",
        # "MVNO 핸드셋 가입자는 41만7천 순증으로 유플러스가 가장 큰 성장을 유지하고 있습니다. 인터넷 순증 마켓쉐어도 성장 추세를 유지하며, 순증 1위를 지속하고 있습니다.",
        # "성과공유회 처음으로 B2B 성과에 대해서도 알려드리겠습니다! 마켓쉐어는 24년 조금 하락했지만, 25년 반등에 성공하였고, 매출은 소폭 상승하여, 2조3802억원을 달성했습니다!",
        # "B2B 성과 중에서도, IDC성과가 두드러지게 나타나고 있는데, 마켓쉐어는 1분기 19.2%에서 4분기 25.3%로 성장하며, 매출은 약 55% 성장하며 4분기 1353억을 달성했습니다!",
        # "모두 박수한번 쳐주세요!",
        # "25년, 전 임직원의 열정과 노력으로 좋은 성과를 만들어 냈습니다. 하지만 경쟁사의 보안이슈가 다시 한번 불거지며 기본기에 대한 중요성은 더욱 대두되었습니다. 구성원 모두 다시 한번 열정 넘치는 노력, 그리고 기본기를 최우선으로 하여 26년에도 성과, 기본기 모두 잡을 수 있는 한 해가 되시길 기대하겠습니다.",
        # "다음 순서로, 컨슈머, 엔터프라이즈 부문 및 기본기 조직인 정보보안, 품질혁신, 안전환경 조직들의 26년 중점 추진과제를 소개해드리겠습니다.",
    ]
    syn_lang_batch = ["Korean"] * len(syn_text_batch)

    # # Reference audio(s)
    # ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    # ref_audio_path_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"

    # ref_audio_single = ref_audio_path_1
    # ref_audio_batch = [ref_audio_path_1, ref_audio_path_2]

    # ref_text_single = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    # ref_text_batch = [
    #     "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    #     "甚至出现交易几乎停滞的情况。",
    # ]

    # # Synthesis targets
    # syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    # syn_lang_single = "Auto"

    # syn_text_batch = [
    #     "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    #     "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    # ]
    # syn_lang_batch = ["Chinese", "English"]

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        non_streaming_mode=True
    )

    for xvec_only in [False, True]:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1: prompt single + synth single, direct
        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
        )
        # run_case(
        #     tts, OUT_DIR, f"case1_promptSingle_synSingle_direct_{mode_tag}",
        #     lambda: tts.generate_voice_clone_with_instruct(
        #         text=syn_text_single,
        #         language=syn_lang_single,
        #         ref_audio=ref_audio_single,
        #         ref_text=ref_text_single,
        #         x_vector_only_mode=xvec_only,
        #         instruct="非常生气。",
        #         **common_gen_kwargs,
        #     ),
        # )

        # # Case 1b: prompt single + synth single, via create_voice_clone_prompt
        # def _case1b():
        #     prompt_items = tts.create_voice_clone_prompt(
        #         ref_audio=ref_audio_single,
        #         ref_text=ref_text_single,
        #         x_vector_only_mode=xvec_only,
        #     )
        #     return tts.generate_voice_clone(
        #         text=syn_text_single,
        #         language=syn_lang_single,
        #         voice_clone_prompt=prompt_items,
        #         **common_gen_kwargs,
        #     )

        # run_case(
        #     tts, OUT_DIR, f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
        #     _case1b,
        # )

        # Case 2: prompt single + synth batch, direct
        # run_case(
        #     tts, OUT_DIR, f"case2_promptSingle_synBatch_direct_{mode_tag}",
        #     lambda: tts.generate_voice_clone(
        #         text=syn_text_batch,
        #         language=syn_lang_batch,
        #         ref_audio=ref_audio_single,
        #         ref_text=ref_text_single,
        #         x_vector_only_mode=xvec_only,
        #         **common_gen_kwargs,
        #     ),
        # )
        # run_case(
        #     tts, OUT_DIR, f"case2_promptSingle_synBatch_direct_{mode_tag}",
        #     lambda: tts.generate_voice_clone_with_instruct(
        #         text=syn_text_batch,
        #         language=syn_lang_batch,
        #         ref_audio=ref_audio_single,
        #         ref_text=ref_text_single,
        #         x_vector_only_mode=xvec_only,
        #         instruct="非常生气。",
        #         **common_gen_kwargs,
        #     ),
        # )

        # # Case 2b: prompt single + synth batch, via create_voice_clone_prompt
        # def _case2b():
        #     prompt_items = tts.create_voice_clone_prompt(
        #         ref_audio=ref_audio_single,
        #         ref_text=ref_text_single,
        #         x_vector_only_mode=xvec_only,
        #     )
        #     return tts.generate_voice_clone(
        #         text=syn_text_batch,
        #         language=syn_lang_batch,
        #         voice_clone_prompt=prompt_items,
        #         **common_gen_kwargs,
        #     )

        # run_case(
        #     tts, OUT_DIR, f"case2_promptSingle_synBatch_promptThenGen_{mode_tag}",
        #     _case2b,
        # )

        # # Case 3: prompt batch + synth batch, direct
        # run_case(
        #     tts, OUT_DIR, f"case3_promptBatch_synBatch_direct_{mode_tag}",
        #     lambda: tts.generate_voice_clone(
        #         text=syn_text_batch,
        #         language=syn_lang_batch,
        #         ref_audio=ref_audio_batch,
        #         ref_text=ref_text_batch,
        #         x_vector_only_mode=[xvec_only, xvec_only],
        #         **common_gen_kwargs,
        #     ),
        # )

        # # Case 3b: prompt batch + synth batch, via create_voice_clone_prompt
        # def _case3b():
        #     prompt_items = tts.create_voice_clone_prompt(
        #         ref_audio=ref_audio_batch,
        #         ref_text=ref_text_batch,
        #         x_vector_only_mode=[xvec_only, xvec_only],
        #     )
        #     return tts.generate_voice_clone(
        #         text=syn_text_batch,
        #         language=syn_lang_batch,
        #         voice_clone_prompt=prompt_items,
        #         **common_gen_kwargs,
        #     )

        # run_case(
        #     tts, OUT_DIR, f"case3_promptBatch_synBatch_promptThenGen_{mode_tag}",
        #     _case3b,
        # )


if __name__ == "__main__":
    main()
