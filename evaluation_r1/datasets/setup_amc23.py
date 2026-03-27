#!/usr/bin/env python3
"""setup_amc23.py — build a Hugging Face Dataset for the 2023 AMC cycle
(10A, 10B, 12A, 12B) and save it with datasets.save_to_disk.

Run:
    python setup_amc23.py              # saves to evaluation_suite/amc23
    python setup_amc23.py --out mydir  # custom output directory
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
from datasets import Dataset, Features, Value

BASE = "https://artofproblemsolving.com/wiki/index.php/"

PAPERS = {
    "10A": "2023_AMC_10A",
    "10B": "2023_AMC_10B",
    "12A": "2023_AMC_12A",
    "12B": "2023_AMC_12B",
}

HEAD_RE = re.compile(r"Problem\s+(\d+)", re.I)


def fetch(url: str) -> BeautifulSoup:
    """Return BeautifulSoup for the given URL."""
    r = requests.get(
        url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (amc23-scraper)"}
    )
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_answers(soup: BeautifulSoup) -> List[str]:
    """Return the 25-element answer key."""
    answers: List[str] = []

    # 1️⃣ ordered list after "Answer Key" heading
    heading = soup.find(
        lambda tag: tag.name in {"h1", "h2", "h3"} and "Answer Key" in tag.get_text()
    )
    if heading:
        ol = heading.find_next("ol")
        if ol:
            answers = [li.get_text(strip=True).split()[-1] for li in ol.find_all("li")]

    # 2️⃣ tolerant plain‑text fallback
    if not answers:
        line_re = re.compile(r"^\s*(\d+)\D+([A-E]|[0-9]+)\s*$")
        for line in soup.get_text("\n").splitlines():
            m = line_re.match(line)
            if m:
                answers.append(m.group(2))

    if len(answers) != 25:
        raise ValueError(f"Expected 25 answers, got {len(answers)}")
    return answers


def extract_problems(soup: BeautifulSoup) -> List[str]:
    """Return the 25 problem statements from an AMC problem page."""
    problems = []
    for h2 in soup.find_all("h2" ) :
        span = h2.find("span", class_="mw-headline")
        if not span or not HEAD_RE.match(span.get_text(strip=True)):
            continue

        parts = []
        for sib in h2.next_siblings:
            if isinstance(sib, Tag) and sib.name == "h2":
                break
            if isinstance(sib, NavigableString):
                txt = sib.strip()
                if txt:
                    parts.append(txt)
            elif isinstance(sib, Tag):
                txt = sib.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)

        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        text = re.split(r"Solution\b", text)[0].strip()
        problems.append(text)

    if len(problems) != 25:
        raise ValueError(f"Expected 25 problems, got {len(problems)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMC 2023 dataset.")
    parser.add_argument(
        "--out",
        default="evaluation_suite/amc23",
        help="Output folder for HF dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    rows = []
    for tag, stem in PAPERS.items():
        print(f"[+] Processing {tag} …", file=sys.stderr)
        problems = extract_problems(fetch(BASE + stem + "_Problems"))
        answers = extract_answers(fetch(BASE + stem + "_Answer_Key"))
        for idx, (p, a) in enumerate(zip(problems, answers), start=1):
            rows.append({"problem": p, "answer": a, "difficulty": f"{tag}-{idx}"})

    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 rows, got {len(rows)}")

    features = Features(
        {
            "problem": Value("string"),
            "answer": Value("string"),
            "difficulty": Value("string"),
        }
    )
    ds = Dataset.from_pandas(pd.DataFrame(rows), features=features)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Saving dataset to {out_dir} …", file=sys.stderr)
    ds.save_to_disk(str(out_dir))
    print(f"[✓] Done → {out_dir}")


if __name__ == "__main__":
    main()
