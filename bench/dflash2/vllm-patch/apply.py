#!/usr/bin/env python3
"""Apply vLLM PR #52816 (DFlash2) onto the 0.27.1 sources extracted from the image.
Every edit asserts its anchor, so a silent mis-apply is impossible."""
import sys, pathlib
R = pathlib.Path.home() / "dflash2-patch" / "patched" / "vllm"
n = 0
def edit(relpath, old, new, count=1):
    global n
    p = R / relpath
    s = p.read_text()
    if new in s and old not in s:
        print(f"  [skip] already applied: {relpath}"); return
    c = s.count(old)
    assert c == count, f"ANCHOR MISMATCH in {relpath}: expected {count} occurrence(s), found {c}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new, count))
    n += 1
    print(f"  [ok]   {relpath}")

# --- A. vllm/config/vllm.py : force the V2 model runner for DFlash2 drafts ---
edit("config/vllm.py",
"""        if self.model_config is not None and self.model_config.is_diffusion:""",
"""        # The DFlash2 candidate selector exists only in the V2 speculator. On V1
        # the same checkpoint drafts through DFlashProposer, which never calls
        # it, so the draft degrades to DFlash1 silently. Force V2 as for dspark.
        if self._is_dflash2_draft():
            return True

        if self.model_config is not None and self.model_config.is_diffusion:""")

edit("config/vllm.py",
"""    def _dflash_needs_multi_kv_group(self) -> bool:""",
'''    def _is_dflash2_draft(self) -> bool:
        """Whether the DFlash draft is a DFlash2 one, by the architecture the
        speculator selects on (v1/worker/gpu/spec_decode/__init__.py)."""
        spec = self.speculative_config
        if spec is None or spec.method != "dflash":
            return False
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            return False
        return "DFlash2DraftModel" in (draft_config.architectures or [])

    def _dflash_needs_multi_kv_group(self) -> bool:''')

# --- B. qwen3_dflash.py : explicit is_causal + subclass hooks ---
edit("model_executor/models/qwen3_dflash.py",
'''    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return override''',
'''    """Resolve explicit causality before falling back to legacy layer defaults."""
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return bool(override)''')

edit("model_executor/models/qwen3_dflash.py",
"""class DFlashQwen3Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(""",
"""class DFlashQwen3Model(nn.Module):
    decoder_layer_cls = DFlashQwen3DecoderLayer

    hf_to_vllm_mapper = WeightsMapper(""")

edit("model_executor/models/qwen3_dflash.py",
"""                DFlashQwen3DecoderLayer(
                    current_vllm_config,""",
"""                self.decoder_layer_cls(
                    current_vllm_config,""")

edit("model_executor/models/qwen3_dflash.py",
"""class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):""",
"""class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    model_cls = DFlashQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):""")

edit("model_executor/models/qwen3_dflash.py",
"""        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,""",
"""        self.model = self.model_cls(
            vllm_config=vllm_config,""")

# --- C. registry.py : register the DFlash2 draft arch ---
edit("model_executor/models/registry.py",
'''    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),''',
'''    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),
    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),''')

# --- D. spec_decode/__init__.py : dispatch to the DFlash2 speculator ---
edit("v1/worker/gpu/spec_decode/__init__.py",
'''    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (''',
'''    if speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (''')

print(f"\napplied {n} edits")
