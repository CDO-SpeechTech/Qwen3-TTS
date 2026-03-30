import os
import glob
import json

metadata_dir = '/data/private/datasets/for_qwen3/AblCase7.txt'
metadata_list = {}
with open(metadata_dir, 'r') as f:
    for line in f:
        _meta = line.strip().split('|')
        metadata_list[_meta[2]] = _meta[-1].strip()


ref_audio = '/data/private/datasets/for_qwen3/F_BOMI/F_BOMI_1_2402.wav'
wav_list = glob.glob('/data/private/datasets/for_qwen3/*/F_BOMI_1_*.wav')
jsonl_list = []
for wav_path in wav_list:
    try:
        txt = metadata_list[wav_path.split('/')[-1].replace('.wav', '')]
    except:
        continue
    jsonl_list.append(
        {"audio":wav_path, "text":txt, "ref_audio":ref_audio}
    )

with open('train_raw.jsonl', 'w', encoding='utf-8') as f:
    for entry in jsonl_list:
        # json.dumps를 통해 딕셔너리를 문자열로 변환 후 줄바꿈(\n) 추가
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
