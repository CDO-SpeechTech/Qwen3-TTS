"""
Streaming 개선사항 반영 후 검증 스크립트.

각 변경사항을 개별적으로 테스트하며, GPU 없이도 import/구조 검증은 가능.
GPU가 있으면 실제 추론까지 검증.

사용법:
    # 전체 검증 (GPU 필요)
    python tests/verify_streaming_changes.py --all

    # import/구조만 검증 (CPU)
    python tests/verify_streaming_changes.py --structure-only

    # 특정 항목만 검증
    python tests/verify_streaming_changes.py --test 1 2 3
"""

import argparse
import sys
import time
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []

def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, bool(passed)))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

def report_skip(name, reason=""):
    results.append((name, None))
    print(f"  [{SKIP}] {name}" + (f" — {reason}" if reason else ""))


# ──────────────────────────────────────────────────────
# Test 1: Helper functions exist (구조 검증)
# ──────────────────────────────────────────────────────
def test_1_helper_functions():
    """헬퍼 함수 존재 여부: _top_k_top_p_filtering, _sample_next_token, _crossfade, _add_ref_code_context"""
    print("\n[Test 1] Helper functions 존재 여부")
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            _top_k_top_p_filtering,
            _sample_next_token,
            _crossfade,
            _add_ref_code_context,
            DEFAULT_BLEND_SAMPLES,
        )
        report("_top_k_top_p_filtering import", True)
        report("_sample_next_token import", True)
        report("_crossfade import", True)
        report("_add_ref_code_context import", True)
        report("DEFAULT_BLEND_SAMPLES == 512", DEFAULT_BLEND_SAMPLES == 512)
    except ImportError as e:
        report("helper functions import", False, str(e))


# ──────────────────────────────────────────────────────
# Test 2: _crossfade 동작 검증
# ──────────────────────────────────────────────────────
def test_2_crossfade():
    """Hann window crossfade 동작 검증"""
    print("\n[Test 2] _crossfade 동작 검증")
    try:
        import numpy as np
        from qwen_tts.core.models.modeling_qwen3_tts import _crossfade

        prev_tail = np.ones(512, dtype=np.float32)
        new_head = np.zeros(512, dtype=np.float32)
        result = _crossfade(prev_tail, new_head)

        report("output length == input length", len(result) == 512)
        report("start ~ 1.0 (fade out prev)", abs(result[0] - 1.0) < 0.01)
        report("end ~ 0.0 (fade in new)", abs(result[-1] - 0.0) < 0.01)
        report("midpoint ~ 0.5 (blend)", abs(result[256] - 0.5) < 0.05)
        report("empty input returns new_head", np.array_equal(_crossfade(np.array([]), new_head), new_head))
    except Exception as e:
        report("_crossfade", False, str(e))


# ──────────────────────────────────────────────────────
# Test 3: _top_k_top_p_filtering, _sample_next_token 동작
# ──────────────────────────────────────────────────────
def test_3_sampling():
    """커스텀 샘플링 함수 동작 검증"""
    print("\n[Test 3] Sampling functions 동작 검증")
    try:
        import torch
        from qwen_tts.core.models.modeling_qwen3_tts import (
            _top_k_top_p_filtering,
            _sample_next_token,
        )

        logits = torch.randn(1, 100)

        # top_k filtering
        filtered = _top_k_top_p_filtering(logits.clone(), top_k=10)
        non_inf = (filtered > float("-inf")).sum().item()
        report("top_k=10 leaves <=10 tokens", non_inf <= 10)

        # top_p filtering
        filtered_p = _top_k_top_p_filtering(logits.clone(), top_p=0.1)
        non_inf_p = (filtered_p > float("-inf")).sum().item()
        report("top_p=0.1 reduces tokens", non_inf_p < 100)

        # sample_next_token
        token = _sample_next_token(logits, temperature=1.0, top_k=10)
        report("sample returns valid shape", token.shape == (1,))
        report("sample returns valid range", 0 <= token.item() < 100)

        # suppress tokens
        token_s = _sample_next_token(logits, temperature=1.0, suppress_tokens=[0, 1, 2, 3, 4])
        report("suppress tokens works", token_s.item() >= 5 or token_s.item() < 0)  # might fail rarely

        # greedy (temperature=0)
        token_g = _sample_next_token(logits, temperature=0)
        report("greedy == argmax", token_g.item() == torch.argmax(logits, dim=-1).item())
    except Exception as e:
        report("sampling functions", False, str(e))


# ──────────────────────────────────────────────────────
# Test 4: _build_talker_inputs 존재 확인
# ──────────────────────────────────────────────────────
def test_4_build_talker_inputs():
    """generate()가 _build_talker_inputs()로 리팩터링되었는지 확인"""
    print("\n[Test 4] _build_talker_inputs 리팩터링 확인")
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        report("_build_talker_inputs method exists",
               hasattr(Qwen3TTSForConditionalGeneration, '_build_talker_inputs'))
        report("generate method exists",
               hasattr(Qwen3TTSForConditionalGeneration, 'generate'))
        report("stream_generate_pcm method exists",
               hasattr(Qwen3TTSForConditionalGeneration, 'stream_generate_pcm'))
        report("batch_stream_generate_pcm method exists",
               hasattr(Qwen3TTSForConditionalGeneration, 'batch_stream_generate_pcm'))
    except Exception as e:
        report("_build_talker_inputs", False, str(e))


# ──────────────────────────────────────────────────────
# Test 5: enable_streaming_optimizations 존재
# ──────────────────────────────────────────────────────
def test_5_streaming_optimizations():
    """torch.compile / CUDA graph 최적화 API 존재 확인"""
    print("\n[Test 5] Streaming optimizations API 확인")
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        report("enable_streaming_optimizations exists",
               hasattr(Qwen3TTSForConditionalGeneration, 'enable_streaming_optimizations'))

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        report("Qwen3TTSModel.enable_streaming_optimizations exists",
               hasattr(Qwen3TTSModel, 'enable_streaming_optimizations'))

        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        report("Tokenizer.enable_streaming_optimizations exists",
               hasattr(Qwen3TTSTokenizer, 'enable_streaming_optimizations'))
        report("Tokenizer.decode_streaming exists",
               hasattr(Qwen3TTSTokenizer, 'decode_streaming'))
        report("Tokenizer.decode_streaming_batch exists",
               hasattr(Qwen3TTSTokenizer, 'decode_streaming_batch'))
    except Exception as e:
        report("streaming optimizations API", False, str(e))


# ──────────────────────────────────────────────────────
# Test 6: Code Predictor fast generation
# ──────────────────────────────────────────────────────
def test_6_fast_codebook():
    """Code Predictor generate_fast / enable_compile 존재 확인"""
    print("\n[Test 6] Fast codebook predictor API 확인")
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
            Qwen3TTSTalkerForConditionalGeneration,
        )
        report("generate_fast exists",
               hasattr(Qwen3TTSTalkerCodePredictorModelForConditionalGeneration, 'generate_fast'))
        report("enable_compile exists",
               hasattr(Qwen3TTSTalkerCodePredictorModelForConditionalGeneration, 'enable_compile'))
        report("enable_fast_codebook_gen exists",
               hasattr(Qwen3TTSTalkerForConditionalGeneration, 'enable_fast_codebook_gen'))
    except Exception as e:
        report("fast codebook API", False, str(e))


# ──────────────────────────────────────────────────────
# Test 7: Decoder streaming methods (tokenizer_12hz)
# ──────────────────────────────────────────────────────
def test_7_decoder_streaming():
    """12Hz Decoder에 streaming 최적화 메서드 존재 확인"""
    print("\n[Test 7] Decoder streaming methods 확인")
    try:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
        # 모델 인스턴스 없이 클래스 레벨에서 메서드 존재 확인
        decoder_cls_name = "Qwen3TTSTokenizerV2Decoder"
        # Decoder는 모델 내부이므로, 모델의 메서드 확인
        report("Qwen3TTSTokenizerV2Model.enable_streaming_optimizations exists",
               hasattr(Qwen3TTSTokenizerV2Model, 'enable_streaming_optimizations'))
        report("Qwen3TTSTokenizerV2Model.decode_streaming exists",
               hasattr(Qwen3TTSTokenizerV2Model, 'decode_streaming'))
    except Exception as e:
        report("decoder streaming methods", False, str(e))


# ──────────────────────────────────────────────────────
# Test 8: High-level streaming API
# ──────────────────────────────────────────────────────
def test_8_highlevel_api():
    """High-level streaming API 존재 확인"""
    print("\n[Test 8] High-level streaming API 확인")
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        report("stream_generate_voice_clone exists",
               hasattr(Qwen3TTSModel, 'stream_generate_voice_clone'))
        report("batch_stream_generate_voice_clone exists",
               hasattr(Qwen3TTSModel, 'batch_stream_generate_voice_clone'))
        # 기존 메서드 보존 확인
        report("generate_voice_clone preserved",
               hasattr(Qwen3TTSModel, 'generate_voice_clone'))
        report("generate_voice_clone_with_instruct preserved",
               hasattr(Qwen3TTSModel, 'generate_voice_clone_with_instruct'))
        report("generate_voice_design preserved",
               hasattr(Qwen3TTSModel, 'generate_voice_design'))
        report("generate_custom_voice preserved",
               hasattr(Qwen3TTSModel, 'generate_custom_voice'))
    except Exception as e:
        report("high-level API", False, str(e))


# ──────────────────────────────────────────────────────
# Test 9: Bug Fix 보존 확인
# ──────────────────────────────────────────────────────
def test_9_bugfix_preserved():
    """Bug Fix 1-3 보존 확인 (forward_finetune의 F.cross_entropy 등)"""
    print("\n[Test 9] Bug Fix 보존 확인")
    try:
        import inspect
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        )

        # Bug Fix 3: forward_finetune에서 F.cross_entropy 직접 사용
        source = inspect.getsource(
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune
        )
        has_cross_entropy = "cross_entropy" in source
        no_loss_function = "self.loss_function" not in source
        report("Bug Fix 3: forward_finetune uses cross_entropy directly", has_cross_entropy)
        report("Bug Fix 3: forward_finetune does NOT use self.loss_function", no_loss_function)

        # from_pretrained hotfix 보존 확인
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        source_fp = inspect.getsource(Qwen3TTSForConditionalGeneration.from_pretrained)
        has_hotfix = "requested_attn_implementation" in source_fp
        report("from_pretrained attn_implementation hotfix preserved", has_hotfix)
    except Exception as e:
        report("bug fix preservation", False, str(e))


# ──────────────────────────────────────────────────────
# Test 10: Multiple EOS 영향 확인 (getattr 버그 수정 포함)
# ──────────────────────────────────────────────────────
def test_10_eos_and_bugfix():
    """generate() 내 Multiple EOS + getattr 버그 수정 확인"""
    print("\n[Test 10] EOS detection + getattr 버그 수정 확인")
    try:
        import inspect
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

        source = inspect.getsource(Qwen3TTSForConditionalGeneration.generate)

        # Multiple EOS
        has_multiple_eos = "eos_ids" in source or "2150" in source
        report("Multiple EOS detection in generate()", has_multiple_eos)

        # getattr 버그 수정 (kwargs.get 사용)
        has_getattr_bug = 'getattr(kwargs, "output_hidden_states"' in source
        has_kwargs_get = 'kwargs.get("output_hidden_states"' in source or 'kwargs.get(' in source
        report("getattr(kwargs,...) bug fixed", not has_getattr_bug,
               "still has getattr(kwargs,...)" if has_getattr_bug else "")
    except Exception as e:
        report("EOS/getattr check", False, str(e))


# ──────────────────────────────────────────────────────
# Test 11: Non-streaming 추론 회귀 테스트 (GPU 필요)
# ──────────────────────────────────────────────────────
def test_11_nonstreaming_regression(model_path):
    """기존 non-streaming 추론이 정상 동작하는지 확인 (GPU 필요)"""
    print("\n[Test 11] Non-streaming 추론 회귀 테스트 (GPU)")
    try:
        import torch
        if not torch.cuda.is_available():
            report_skip("non-streaming regression", "CUDA not available")
            return

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        # 간단한 텍스트로 추론
        t0 = time.time()
        wavs, sr = tts.generate_voice_clone(
            text="Hello, this is a test.",
            language="English",
            ref_audio="kuklina-1.wav",
            ref_text="Hello world",
            x_vector_only_mode=True,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
        )
        elapsed = time.time() - t0

        report("generate_voice_clone returns wavs", len(wavs) > 0)
        report("wav has samples", wavs[0].shape[0] > 0)
        report("sample_rate valid", sr > 0)
        report(f"inference time ({elapsed:.2f}s)", elapsed < 30, f"{elapsed:.2f}s")
        print(f"    wav shape: {wavs[0].shape}, sr: {sr}")
    except Exception as e:
        report("non-streaming regression", False, traceback.format_exc())


# ──────────────────────────────────────────────────────
# Test 12: Streaming 추론 테스트 (GPU 필요)
# ──────────────────────────────────────────────────────
def test_12_streaming_inference(model_path):
    """스트리밍 추론이 정상 동작하는지 확인 (GPU 필요)"""
    print("\n[Test 12] Streaming 추론 테스트 (GPU)")
    try:
        import torch
        if not torch.cuda.is_available():
            report_skip("streaming inference", "CUDA not available")
            return

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        chunks = []
        t0 = time.time()
        first_chunk_time = None
        for chunk, sr in tts.stream_generate_voice_clone(
            text="Hello, this is a streaming test.",
            language="English",
            ref_audio="kuklina-1.wav",
            ref_text="Hello world",
            x_vector_only_mode=True,
            emit_every_frames=8,
            decode_window_frames=80,
            max_frames=256,
            do_sample=True,
            temperature=0.9,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
            chunks.append(chunk)
        total_time = time.time() - t0

        report("streaming yields chunks", len(chunks) > 0)
        report("each chunk is numpy array", all(hasattr(c, 'shape') for c in chunks))
        report(f"first chunk latency ({first_chunk_time:.3f}s)", first_chunk_time < 5)
        report(f"total chunks: {len(chunks)}", True)
        print(f"    first_chunk: {first_chunk_time:.3f}s, total: {total_time:.2f}s, chunks: {len(chunks)}")
    except Exception as e:
        report("streaming inference", False, traceback.format_exc())


# ──────────────────────────────────────────────────────
# Test 13: Multiple EOS 실제 영향 확인 (GPU 필요)
# ──────────────────────────────────────────────────────
def test_13_eos_impact(model_path):
    """Multiple EOS가 기존 추론 길이에 미치는 영향 확인 (GPU 필요)"""
    print("\n[Test 13] Multiple EOS 실제 영향 확인 (GPU)")
    try:
        import torch
        if not torch.cuda.is_available():
            report_skip("EOS impact", "CUDA not available")
            return

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        # 동일 시드로 여러 번 추론하여 길이 비교
        torch.manual_seed(42)
        wavs1, sr1 = tts.generate_voice_clone(
            text="The quick brown fox jumps over the lazy dog.",
            language="English",
            ref_audio="kuklina-1.wav",
            ref_text="Hello world",
            x_vector_only_mode=True,
            max_new_tokens=1024,
            do_sample=False,  # greedy for reproducibility
            temperature=0.0,
        )

        len1 = wavs1[0].shape[0]
        report(f"generated wav length: {len1} samples ({len1/sr1:.2f}s)", len1 > 0)
        report("wav not suspiciously short (<0.5s)", len1 / sr1 > 0.5,
               f"duration: {len1/sr1:.2f}s")
        print(f"    wav length: {len1} samples = {len1/sr1:.2f}s at {sr1}Hz")
    except Exception as e:
        report("EOS impact", False, traceback.format_exc())


# ──────────────────────────────────────────────────────
# Test 14: torch.compile 최적화 검증 (GPU 필요)
# ──────────────────────────────────────────────────────
def test_14_compile_optimization(model_path):
    """torch.compile 최적화 활성화 + 스트리밍 속도 비교 (GPU 필요)"""
    print("\n[Test 14] torch.compile 최적화 검증 (GPU)")
    try:
        import torch
        if not torch.cuda.is_available():
            report_skip("compile optimization", "CUDA not available")
            return

        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        import numpy as np

        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        test_kwargs = dict(
            text="Testing optimization with streaming generation.",
            language="English",
            ref_audio="kuklina-1.wav",
            ref_text="Hello world",
            x_vector_only_mode=True,
            emit_every_frames=8,
            decode_window_frames=80,
            max_frames=256,
            do_sample=True,
            temperature=0.9,
        )

        # Baseline (no optimization)
        t0 = time.time()
        chunks_baseline = list(tts.stream_generate_voice_clone(**test_kwargs))
        time_baseline = time.time() - t0

        # Enable optimizations
        tts.enable_streaming_optimizations(decode_window_frames=80)

        # Warmup
        _ = list(tts.stream_generate_voice_clone(**test_kwargs))

        # Optimized
        t0 = time.time()
        chunks_optimized = list(tts.stream_generate_voice_clone(**test_kwargs))
        time_optimized = time.time() - t0

        speedup = time_baseline / max(time_optimized, 0.001)
        report(f"baseline: {time_baseline:.2f}s, optimized: {time_optimized:.2f}s", True)
        report(f"speedup: {speedup:.2f}x", speedup > 0.8,  # at least not slower
               f"{'faster' if speedup > 1 else 'slower'}")
        print(f"    baseline: {time_baseline:.2f}s ({len(chunks_baseline)} chunks)")
        print(f"    optimized: {time_optimized:.2f}s ({len(chunks_optimized)} chunks)")
        print(f"    speedup: {speedup:.2f}x")
    except Exception as e:
        report("compile optimization", False, traceback.format_exc())


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Streaming changes verification")
    parser.add_argument("--all", action="store_true", help="Run all tests including GPU")
    parser.add_argument("--structure-only", action="store_true", help="Only run structure/import tests (no GPU)")
    parser.add_argument("--test", nargs="+", type=int, help="Run specific test numbers")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Model path for GPU tests")
    args = parser.parse_args()

    structure_tests = {
        1: test_1_helper_functions,
        2: test_2_crossfade,
        3: test_3_sampling,
        4: test_4_build_talker_inputs,
        5: test_5_streaming_optimizations,
        6: test_6_fast_codebook,
        7: test_7_decoder_streaming,
        8: test_8_highlevel_api,
        9: test_9_bugfix_preserved,
        10: test_10_eos_and_bugfix,
    }
    gpu_tests = {
        11: lambda: test_11_nonstreaming_regression(args.model_path),
        12: lambda: test_12_streaming_inference(args.model_path),
        13: lambda: test_13_eos_impact(args.model_path),
        14: lambda: test_14_compile_optimization(args.model_path),
    }

    if args.test:
        tests_to_run = {k: v for k, v in {**structure_tests, **gpu_tests}.items() if k in args.test}
    elif args.structure_only:
        tests_to_run = structure_tests
    elif args.all:
        tests_to_run = {**structure_tests, **gpu_tests}
    else:
        tests_to_run = structure_tests  # default: structure only

    print("=" * 60)
    print("Streaming Changes Verification")
    print("=" * 60)

    for num, test_fn in sorted(tests_to_run.items()):
        try:
            test_fn()
        except Exception as e:
            report(f"Test {num} (unexpected error)", False, str(e))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p is True)
    failed = sum(1 for _, p in results if p is False)
    skipped = sum(1 for _, p in results if p is None)
    total = len(results)
    print(f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        print("\nFailed tests:")
        for name, p in results:
            if p is False:
                print(f"  - {name}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
