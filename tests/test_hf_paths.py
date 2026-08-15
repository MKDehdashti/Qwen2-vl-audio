"""Static invariants for the Hugging Face checkpoint paths in src/.

These are deliberately dependency-free: they parse the source with `ast` rather
than importing it, because src/train.py pulls in torch, trl and wandb at module
level and CI should not need a 2 GB install to catch a renamed folder.

They guard one specific failure class, which has now occurred twice. The repo
was reorganized from a flat namespace (``avqa_stage1_whisper32/``) into a tree
(``avqa/ablations/whisper32/stage1/``). Both times, a call site was missed
because it did not look like the others:

  * ``setup_avqa_whisper_fullres_notts_stage2`` takes ``subfolder=None`` and
    *computes* the path in the body, so it never matched a grep for
    ``subfolder='avqa``.
  * ``eval_asr_post_avqa.py`` held its paths in module-level constants, in a
    different file entirely.

A wrong path here does not raise. ``from_pretrained`` on the repo root returns
the finished fine-tuned model instead of the ASR base, and training silently
starts from the wrong weights.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
TRAIN_PY = SRC / "train.py"

# The tree that actually exists on the Hub. Anything outside it is a typo.
VALID_PREFIXES = ("asr/", "avqa/")

# Literals from the pre-reorganization flat namespace. None of these may appear
# as a *path*; they are still legal as W&B run names (stage_name=...), which is
# why the check below looks only at path-carrying keyword arguments.
STALE_PATH_RE = re.compile(r"^(avqa_stage[12]_|lora_stage[23]$|avqa_init$)")


def _load(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _extract_function(path: pathlib.Path, name: str):
    """Return a callable for one top-level function, without importing the module."""
    tree = _load(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns: dict = {}
            # carry along any module-level assignments the function closes over
            for prior in tree.body:
                if isinstance(prior, ast.Assign):
                    try:
                        exec(compile(ast.Module([prior], []), "<x>", "exec"), ns)
                    except Exception:
                        pass  # ignore anything needing an import
            exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {path.name}")


def _path_keyword_literals(path: pathlib.Path):
    """Yield (lineno, value) for every string literal passed as a path kwarg."""
    for node in ast.walk(_load(path)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg in ("path_in_repo", "subfolder") and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    yield kw.value.lineno, kw.value.value


# every experiment tag that has ever been pushed, and where it belongs
EXPECTED = {
    ("whisper_fullres_v2", 1): "avqa/headline/stage1",
    ("whisper_fullres_v2", 2): "avqa/headline/stage2_qproj_tuned",
    ("whisper_fullres_v3", 1): "avqa/headline/stage1",
    ("whisper_fullres_v3", 2): "avqa/headline/stage2_qproj_frozen",
    ("whisper_fullres_v3_seed1234", 1): "avqa/seeds/seed1234/stage1",
    ("whisper_fullres_v3_seed2026", 2): "avqa/seeds/seed2026/stage2",
    ("panns8", 1): "avqa/ablations/panns8/stage1",
    ("panns32", 2): "avqa/ablations/panns32/stage2",
    ("whisper32", 1): "avqa/ablations/whisper32/stage1",
    ("whisper32_full", 2): "avqa/ablations/whisper32_full/stage2",
    ("whisper_fullres_notts", 1): "avqa/ablations/whisper_fullres_notts/stage1",
    ("whisper_fullres_notts_matched", 1): "avqa/ablations/whisper_fullres_notts_matched/stage1",
}


@pytest.mark.parametrize("key,expected", sorted(EXPECTED.items()))
def test_avqa_hf_path_maps_each_tag(key, expected):
    """The headline and seed tags are special-cased; everything else is an ablation."""
    avqa_hf_path = _extract_function(TRAIN_PY, "avqa_hf_path")
    tag, stage = key
    assert avqa_hf_path(tag, stage) == expected


def test_headline_stage1_is_shared_between_v2_and_v3():
    """v3's Stage 2 trained from v2's Stage 1, so both must resolve to one folder.

    If these ever diverge, the v3 reproduction silently trains from the wrong
    Stage-1 checkpoint.
    """
    avqa_hf_path = _extract_function(TRAIN_PY, "avqa_hf_path")
    assert avqa_hf_path("whisper_fullres_v2", 1) == avqa_hf_path("whisper_fullres_v3", 1)


def test_the_two_stage2_variants_do_not_collide():
    """Frozen and tuned question-projector runs must not overwrite each other."""
    avqa_hf_path = _extract_function(TRAIN_PY, "avqa_hf_path")
    assert avqa_hf_path("whisper_fullres_v2", 2) != avqa_hf_path("whisper_fullres_v3", 2)


@pytest.mark.parametrize("py", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_stale_flat_namespace_paths(py):
    """No path kwarg may still point at the pre-reorganization flat namespace."""
    bad = [(ln, v) for ln, v in _path_keyword_literals(py) if STALE_PATH_RE.match(v)]
    assert not bad, f"{py.name}: stale flat HF path(s): {bad}"


def _is_root_level_file(value: str) -> bool:
    """e.g. path_in_repo='config.json' - a single file at the repo root, not a checkpoint."""
    return "/" not in value and pathlib.PurePosixPath(value).suffix != ""


@pytest.mark.parametrize("py", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_path_literals_live_in_the_known_tree(py):
    """Every hardcoded checkpoint path sits under asr/ or avqa/.

    Root-level single files (config.json, README.md) are exempt - those are
    legitimately uploaded to the repo root.
    """
    bad = [(ln, v) for ln, v in _path_keyword_literals(py)
           if not v.startswith(VALID_PREFIXES) and not _is_root_level_file(v)]
    assert not bad, f"{py.name}: path outside the asr/ + avqa/ tree: {bad}"


def test_avqa_runs_initialize_from_the_asr_merge_not_the_repo_root():
    """The repo root is the *finished* AVQA model.

    Loading it bare does not raise - it just starts training from fine-tuned
    weights. Every init_avqa_*_model must pass the ASR merge subfolder.
    """
    tree = _load(TRAIN_PY)
    offenders = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and re.fullmatch(r"init_avqa_\w*model", node.name)):
            continue
        src = ast.get_source_segment(TRAIN_PY.read_text(), node) or ""
        if "ASR_MERGE_SUBFOLDER" not in src:
            offenders.append(node.name)
    assert not offenders, f"loads the bare repo root: {offenders}"


def test_asr_merge_subfolder_constant_is_correct():
    for node in _load(TRAIN_PY).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ASR_MERGE_SUBFOLDER":
            assert node.value.value == "asr/merged_stage2"
            return
    raise AssertionError("ASR_MERGE_SUBFOLDER is not defined")


def test_push_helpers_report_the_path_they_actually_used():
    """Push messages must interpolate the variable, not repeat a literal.

    Twelve of these drifted: path_in_repo was updated to the new tree while the
    print() below it still announced the old flat folder.
    """
    text = TRAIN_PY.read_text()
    bad = [ln for ln, line in enumerate(text.splitlines(), 1)
           if "pushed to {repo_id}/" in line and "{hf_subfolder}" not in line]
    assert not bad, f"push message hardcodes a path instead of using hf_subfolder, lines: {bad}"
