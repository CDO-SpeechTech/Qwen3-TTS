"""
Test streaming text input with character-by-character yielding.

Demonstrates feeding text to the TTS model one character at a time
via a Python generator, simulating real-time text arrival (e.g., from LLM output).

Usage:
    cd Qwen3-TTS
    python examples/test_streaming_text_input.py
"""

import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision('high')


def char_by_char(text: str, delay: float = 0.0):
    """1글자씩 yield하는 제너레이터. delay로 LLM 출력 속도를 시뮬레이션."""
    for ch in text:
        if delay > 0:
            time.sleep(delay)
        yield ch


def main():
    total_start = time.time()

    # ── 모델 로드 ──
    print("Loading model...")
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"[{time.time() - start:.2f}s] Model loaded")

    # ── 레퍼런스 오디오 ──
    ref_audio_path = "kuklina-1.wav"
    ref_text = (
        "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? У него ведь куча наград по "
        "боевым искусствам. Кэти рассказывала, правда, Лео? Понимаешь кого ты побила, Лая? "
        "Только потрогай эти мышцы... Не знала, что у тебя такой классный котик. Рожденная луной. "
        "Лай всегда откопает что-нибудь этакое. Да, жаль только, что занимает почти всё её время. "
        "Не понимаю, почему эта рухлядь не может подождать, пока ты проведешь время с сестрой."
    )

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )

    # ── 한국어 테스트 문장 ──
    test_text = "안녕하세요, 저는 음성 합성 테스트를 진행하고 있습니다. 한 글자씩 스트리밍으로 텍스트가 입력되고 있어요."

    # ============================================================
    # Test 1: 일반 생성 (baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 1: 일반 생성 (baseline)")
    print("=" * 60)

    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=test_text,
        language="Korean",
        voice_clone_prompt=voice_clone_prompt,
    )
    baseline_time = time.time() - start
    baseline_audio = wavs[0]
    baseline_duration = len(baseline_audio) / sr
    sf.write("output_baseline_ko.wav", baseline_audio, sr)
    print(f"Time: {baseline_time:.2f}s | Audio: {baseline_duration:.2f}s | RTF: {baseline_time / baseline_duration:.2f}")

    # ============================================================
    # Test 2: 스트리밍 (문자열 한번에 전달)
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 2: 스트리밍 (텍스트 한번에 전달)")
    print("=" * 60)

    start = time.time()
    chunks = []
    first_chunk_time = None

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=test_text,
        language="Korean",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
        decode_window_frames=80,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"  First chunk: {first_chunk_time:.2f}s ({len(chunk)} samples)")

    streaming_time = time.time() - start
    streaming_audio = np.concatenate(chunks)
    streaming_duration = len(streaming_audio) / chunk_sr
    sf.write("output_streaming_ko.wav", streaming_audio, chunk_sr)
    print(f"Time: {streaming_time:.2f}s | Audio: {streaming_duration:.2f}s | Chunks: {len(chunks)} | RTF: {streaming_time / streaming_duration:.2f}")

    # ============================================================
    # Test 3: 스트리밍 텍스트 입력 (1글자씩 yield)
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 3: 스트리밍 텍스트 입력 (char-by-char)")
    print("=" * 60)
    print(f"  Input: \"{test_text}\"")
    print(f"  총 {len(test_text)}글자를 1글자씩 yield")

    start = time.time()
    chunks = []
    first_chunk_time = None

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=char_by_char(test_text, delay=0.0),  # 글자 단위 제너레이터
        language="Korean",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
        decode_window_frames=80,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"  First chunk: {first_chunk_time:.2f}s ({len(chunk)} samples)")

    text_stream_time = time.time() - start
    text_stream_audio = np.concatenate(chunks)
    text_stream_duration = len(text_stream_audio) / chunk_sr
    sf.write("output_text_streaming_ko.wav", text_stream_audio, chunk_sr)
    print(f"Time: {text_stream_time:.2f}s | Audio: {text_stream_duration:.2f}s | Chunks: {len(chunks)} | RTF: {text_stream_time / text_stream_duration:.2f}")

    # ============================================================
    # Test 4: LLM 출력 시뮬레이션 (글자 사이 50ms 딜레이)
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 4: LLM 출력 시뮬레이션 (char-by-char, 50ms delay)")
    print("=" * 60)

    start = time.time()
    chunks = []
    first_chunk_time = None

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=char_by_char(test_text, delay=0.05),  # 50ms per char ≈ 초당 20자
        language="Korean",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
        decode_window_frames=80,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"  First chunk: {first_chunk_time:.2f}s ({len(chunk)} samples)")

    llm_sim_time = time.time() - start
    llm_sim_audio = np.concatenate(chunks)
    llm_sim_duration = len(llm_sim_audio) / chunk_sr
    sf.write("output_llm_sim_ko.wav", llm_sim_audio, chunk_sr)
    print(f"Time: {llm_sim_time:.2f}s | Audio: {llm_sim_duration:.2f}s | Chunks: {len(chunks)} | RTF: {llm_sim_time / llm_sim_duration:.2f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<35} {'Time':>8} {'Audio':>8} {'1st Chunk':>10} {'RTF':>6}")
    print("-" * 70)
    print(f"{'Baseline (non-streaming)':<35} {baseline_time:>7.2f}s {baseline_duration:>7.2f}s {'N/A':>10} {baseline_time / baseline_duration:>6.2f}")
    print(f"{'Streaming (full text)':<35} {streaming_time:>7.2f}s {streaming_duration:>7.2f}s {first_chunk_time:>9.2f}s {streaming_time / streaming_duration:>6.2f}")
    print(f"{'Streaming (char-by-char)':<35} {text_stream_time:>7.2f}s {text_stream_duration:>7.2f}s {'—':>10} {text_stream_time / text_stream_duration:>6.2f}")
    print(f"{'Streaming (char + 50ms delay)':<35} {llm_sim_time:>7.2f}s {llm_sim_duration:>7.2f}s {'—':>10} {llm_sim_time / llm_sim_duration:>6.2f}")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")


if __name__ == "__main__":
    main()
