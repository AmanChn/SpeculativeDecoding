"""Greedy, lossless draft-model speculative decoding. Pretrained weights only, no training."""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain how binary search works and give a Python implementation.",
    "Write a Python class implementing a stack with push, pop and peek.",
    "Explain the difference between a process and a thread.",
    "Write a SQL query to find the second highest salary in an Employee table.",
]


@torch.no_grad()
def baseline(target, ids, max_new, eos):
    cache = DynamicCache()
    logits = target(ids, past_key_values=cache, use_cache=True).logits[:, -1]
    nxt = logits.argmax(-1, keepdim=True)
    out = [nxt]
    for _ in range(max_new - 1):
        if nxt.item() == eos:
            break
        logits = target(nxt, past_key_values=cache, use_cache=True).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        out.append(nxt)
    return trim(torch.cat(out, 1), max_new, eos), {}


@torch.no_grad()
def speculative(target, draft, ids, max_new, eos, k):
    tc, dc = DynamicCache(), DynamicCache()
    logits = target(ids, past_key_values=tc, use_cache=True).logits[:, -1]
    draft(ids, past_key_values=dc, use_cache=True)
    seq = torch.cat([ids, logits.argmax(-1, keepdim=True)], 1)
    plen = ids.shape[1]
    rounds = accepted = 0

    # Invariant: both caches hold seq[:-1]
    while seq.shape[1] - plen < max_new and not (seq[0, plen:] == eos).any():
        n_tok = seq.shape[1]

        inp = seq[:, dc.get_seq_length():]
        drafts = []
        for _ in range(k):
            nxt = draft(inp, past_key_values=dc, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
            drafts.append(nxt)
            inp = nxt
        d = torch.cat(drafts, 1)

        v = torch.cat([seq[:, -1:], d], 1)
        t = target(v, past_key_values=tc, use_cache=True).logits.argmax(-1)

        n = int((t[:, :k] == d).int().cumprod(1).sum())
        seq = torch.cat([seq, t[:, : n + 1]], 1)

        tc.crop(n_tok + n)
        dc.crop(n_tok + n)
        rounds += 1
        accepted += n

    out = trim(seq[:, plen:], max_new, eos)
    return out, {"rounds": rounds, "accepted": accepted, "drafted": rounds * k}


def trim(tokens, max_new, eos):
    tokens = tokens[:, :max_new]
    hit = (tokens[0] == eos).nonzero()
    return tokens[:, : hit[0].item() + 1] if len(hit) else tokens


def timed(fn, *args):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, stats = fn(*args)
    torch.cuda.synchronize()
    return out, stats, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-32B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.target)
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    target = AutoModelForCausalLM.from_pretrained(a.target, **kw).eval()
    draft = AutoModelForCausalLM.from_pretrained(a.draft, **kw).eval()
    eos = tok.eos_token_id

    def encode(p):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return tok(text, return_tensors="pt").input_ids.cuda()

    batch = [encode(p) for p in PROMPTS]

    for i in range(a.warmup):
        baseline(target, batch[0], 32, eos)
        for k in a.k:
            speculative(target, draft, batch[0], 32, eos, k)

    base_tok = base_time = 0
    base_out = []
    for ids in batch:
        out, _, dt = timed(baseline, target, ids, a.max_new, eos)
        base_out.append(out)
        base_tok += out.shape[1]
        base_time += dt
    base_tps = base_tok / base_time
    print(f"baseline: {base_tps:.1f} tok/s")

    print(f"{'k':>3} {'tok/s':>8} {'speedup':>8} {'accept':>8} {'tok/round':>10} {'match':>6}")
    for k in a.k:
        n_tok = t_sum = rounds = acc = drafted = 0
        match = 0
        for ids, ref in zip(batch, base_out):
            out, s, dt = timed(speculative, target, draft, ids, a.max_new, eos, k)
            n_tok += out.shape[1]
            t_sum += dt
            rounds += s["rounds"]
            acc += s["accepted"]
            drafted += s["drafted"]
            match += int(out.shape == ref.shape and bool((out == ref).all()))
        tps = n_tok / t_sum
        print(f"{k:>3} {tps:>8.1f} {tps / base_tps:>7.2f}x {acc / drafted:>8.2%} "
              f"{(acc + rounds) / rounds:>10.2f} {match:>3}/{len(batch)}")

    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
