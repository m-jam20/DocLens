"""Measure retrieval quality: does the right page show up in the top-k chunks?

Each question in eval/questions.json names the source PDF and the PDF page
range(s) that actually answer it; a question can list several ranges when the
manual covers the topic in more than one place. A retrieved chunk counts as a
hit when it comes from that PDF and overlaps any expected range. Only
retrieval is scored (no LLM call), so a full run takes seconds.

Uses the same retrieval as the app (vector search + reranker); pass
--no-rerank to score plain vector search for comparison.
"""

import argparse
import json

import query


def is_hit(meta, expected):
    return meta["source"] == expected["source"] and any(
        meta["page_start"] <= end and meta["page_end"] >= start
        for start, end in expected["pages"]
    )


def fmt_ranges(ranges):
    return ", ".join(f"p.{s}" if s == e else f"p.{s}-{e}" for s, e in ranges)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("-k", type=int, default=query.TOP_K, help="chunks retrieved per question")
    parser.add_argument("--no-rerank", action="store_true", help="vector search only")
    parser.add_argument("--json", metavar="PATH", help="also write results to this file")
    args = parser.parse_args()

    with open(args.questions) as f:
        cases = json.load(f)

    embedder, collection = query.load_embedder(), query.load_collection()
    reranker = None if args.no_rerank else query.load_reranker()

    rows = []
    for case in cases:
        hits = query.retrieve(embedder, collection, case["question"], k=args.k, reranker=reranker)
        metas = [h["meta"] for h in hits]
        rank = next((i for i, m in enumerate(metas, start=1) if is_hit(m, case)), None)
        rows.append({**case, "rank": rank, "top1": query.cite(metas[0])})

    print(f"\n{'rank':>4}  question")
    for r in rows:
        mark = f"{r['rank']:>4}" if r["rank"] else "MISS"
        line = f"{mark}  {r['question']}"
        if r["rank"] != 1:
            expected = f"{r['source'].removesuffix('.pdf')} {fmt_ranges(r['pages'])}"
            line += f"\n      expected {expected}, top hit was {r['top1']}"
        print(line)

    n = len(rows)
    metrics = {
        "questions": n,
        "k": args.k,
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "hit@3": sum(r["rank"] is not None and r["rank"] <= 3 for r in rows) / n,
        f"hit@{args.k}": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
    }

    mode = "vector search only" if args.no_rerank else "vector search + reranker"
    print(f"\n{n} questions, k={args.k}, {mode}")
    for name in dict.fromkeys(("hit@1", "hit@3", f"hit@{args.k}", "mrr")):
        value = metrics[name]
        print(f"  {name:<6} {value:.2f}" if name == "mrr" else f"  {name:<6} {value:.0%}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"metrics": metrics, "results": rows}, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
