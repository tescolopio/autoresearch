#!/usr/bin/env python3
"""Minimal stdin-to-stdout local provider stub for provider_benchmark.py examples."""

import sys


def main():
    prompt = sys.stdin.read().strip()
    if "17 sheep" in prompt:
        print("9")
        return
    if "27 * 14" in prompt:
        print("27 * 14 = 378")
        return
    print("1. Run the local model. 2. Run the API model. 3. Compare latency, tokens/sec, and answer quality.")


if __name__ == "__main__":
    main()