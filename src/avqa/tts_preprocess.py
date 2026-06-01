# src/avqa/tts_preprocess.py
# One-time preprocessing: convert all MUSIC-AVQA questions to speech via edge-tts.
# Saves {question_id}.wav for each question; duplicate texts share one audio file via symlinks.
#
# Usage:
#   python src/avqa/tts_preprocess.py
#   python src/avqa/tts_preprocess.py --out_dir music_avqa_dataset/data/tts_questions --concurrency 20
#
# Requires: edge-tts  (pip install edge-tts)
# Runtime: ~10-15 min (only 2,815 unique texts out of 45,629 questions; rest are symlinked)
# Resume-safe: skips existing files and symlinks.

import argparse
import asyncio
import json
import sys
from pathlib import Path

import edge_tts
from tqdm import tqdm

# fill_placeholders lives in dataset.py (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avqa.dataset import fill_placeholders  # noqa: E402

VOICE      = "en-US-AriaNeural"
JSON_FILES = [
    "music_avqa_dataset/data/json/avqa-train.json",
    "music_avqa_dataset/data/json/avqa-val.json",
    "music_avqa_dataset/data/json/avqa-test.json",
]


def load_questions(json_files: list[str]) -> dict[int, str]:
    """Return {question_id: question_text} for all splits (placeholders filled)."""
    questions = {}
    for path in json_files:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        for item in json.load(open(p)):
            qid = item["question_id"]
            if qid not in questions:
                text = fill_placeholders(item["question_content"],
                                         item.get("templ_values", "[]"))
                questions[qid] = text
    return questions


def build_dedup_plan(questions: dict[int, str]) -> tuple[dict[int, str], dict[int, int]]:
    """
    Split questions into:
      canonical: {qid: text}  — one qid per unique text (TTS these)
      symlinks:  {qid: canonical_qid}  — remaining qids point to their canonical
    """
    text_to_canonical: dict[str, int] = {}
    canonical: dict[int, str] = {}
    symlinks: dict[int, int] = {}

    for qid, text in questions.items():
        if text not in text_to_canonical:
            text_to_canonical[text] = qid
            canonical[qid] = text
        else:
            symlinks[qid] = text_to_canonical[text]

    return canonical, symlinks


async def tts_one(qid: int, text: str, out_path: Path,
                  sem: asyncio.Semaphore, errors: list):
    """Synthesise one question. Skips if file already exists."""
    if out_path.exists():
        return
    async with sem:
        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(str(out_path))
        except Exception as e:
            errors.append(qid)
            tqdm.write(f"  ERROR qid={qid}: {e}")


def create_symlinks(symlinks: dict[int, int], out_dir: Path) -> int:
    """Create {qid}.wav → {canonical_qid}.wav symlinks. Returns count created."""
    created = 0
    for qid, canonical_qid in symlinks.items():
        link = out_dir / f"{qid}.wav"
        if link.exists() or link.is_symlink():
            continue
        target = out_dir / f"{canonical_qid}.wav"
        if not target.exists():
            continue  # canonical not yet synthesised; will be created on re-run
        link.symlink_to(target.name)  # relative symlink
        created += 1
    return created


async def run(questions: dict[int, str], out_dir: Path, concurrency: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical, symlinks = build_dedup_plan(questions)
    print(f"Total questions    : {len(questions)}")
    print(f"Unique texts (TTS) : {len(canonical)}")
    print(f"Symlinked dupes    : {len(symlinks)}")

    todo = {qid: txt for qid, txt in canonical.items()
            if not (out_dir / f"{qid}.wav").exists()}
    print(f"Already done       : {len(canonical) - len(todo)}")
    print(f"To synthesise      : {len(todo)}")

    if todo:
        sem    = asyncio.Semaphore(concurrency)
        errors = []

        tasks = [
            tts_one(qid, txt, out_dir / f"{qid}.wav", sem, errors)
            for qid, txt in todo.items()
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="TTS"):
            await coro

        done = len(todo) - len(errors)
        print(f"\nSynthesised: {done}  Errors: {len(errors)}")
        if errors:
            err_file = out_dir / "errors.txt"
            err_file.write_text("\n".join(str(e) for e in errors))
            print(f"Failed question IDs saved to {err_file}")
            print("Re-run the script to retry failed questions.")
    else:
        print("All unique texts already synthesised.")

    n_links = create_symlinks(symlinks, out_dir)
    print(f"Symlinks created   : {n_links}")
    total_files = len(list(out_dir.glob("*.wav")))
    print(f"Total .wav files   : {total_files} / {len(questions)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     default="music_avqa_dataset/data/tts_questions")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max parallel edge-tts requests")
    parser.add_argument("--json_dir",    default=None,
                        help="Override JSON directory (default: inferred from JSON_FILES)")
    args = parser.parse_args()

    json_files = JSON_FILES
    if args.json_dir:
        json_files = [str(Path(args.json_dir) / f) for f in
                      ["avqa-train.json", "avqa-val.json", "avqa-test.json"]]

    questions = load_questions(json_files)
    asyncio.run(run(questions, Path(args.out_dir), args.concurrency))


if __name__ == "__main__":
    main()
