"""Check the speaker script's stated per-slide timings against its actual word count.

Only text inside \\begin{say}...\\end{say} is spoken; \\cue{} is a stage direction and everything
after the last slide is Q&A prep, so neither counts. Exits nonzero if the full script overruns
LIMIT, which is what makes "under 10 minutes" an auditable claim rather than an aspiration.

  python tools/script_timing.py [--wpm 150] [--limit 600] [--propose]

--propose prints cumulative timings derived from the real word counts, for pasting back into
the \\slide{}{}{} headers. Slides listed in OPTIONAL are excluded from the 10-minute path.
"""
import argparse
import re
import sys

SRC = "slides/script.tex"
# Detail slides the team explicitly said could go ("good to skip details"). Cut these first.
OPTIONAL = {5, 6}


def blocks(path):
    """Yield (number, title, stated_budget_seconds, spoken_word_count) per slide.

    Two header forms: "{m:ss--m:ss}" for slides on the 10-minute path, and
    "{optional +m:ss}" for the detail slides that get cut first.
    """
    txt = open(path).read()
    parts = re.split(r"\\slide\{(\d+)\}\{([^}]*)\}"
                     r"\{(?:(\d+):(\d\d)--(\d+):(\d\d)|optional \+(\d+):(\d\d))\}", txt)
    for i in range(1, len(parts), 9):
        n, title, m1, s1, m2, s2, om, os_ = parts[i:i + 8]
        spoken = " ".join(re.findall(r"\\begin\{say\}(.*?)\\end\{say\}",
                                     parts[i + 8], re.S))
        spoken = re.sub(r"%.*", "", spoken)
        spoken = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", spoken)
        spoken = re.sub(r"[{}$~&\\]", " ", spoken)
        if om is not None:
            budget = int(om) * 60 + int(os_)
        else:
            budget = (int(m2) * 60 + int(s2)) - (int(m1) * 60 + int(s1))
        yield int(n), title, budget, len(spoken.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wpm", type=int, default=150)
    p.add_argument("--limit", type=int, default=600, help="seconds the talk must fit in")
    p.add_argument("--propose", action="store_true")
    p.add_argument("--src", default=SRC)
    a = p.parse_args()

    rows = list(blocks(a.src))
    if not rows:
        sys.exit("no \\slide{}{}{} headers found in %s" % a.src)

    full = core = 0
    print("%-4s %-42s %5s %6s %6s" % ("#", "slide", "budg", "words", "need"))
    for n, title, budget, words in rows:
        need = words / a.wpm * 60
        print("%-4s %-42s %5d %6d %6.0f%s"
              % (n, title[:42], budget, words, need,
                 "   (optional)" if n in OPTIONAL else ""))
        full += words
        if n not in OPTIONAL:
            core += words

    print("-" * 68)
    for label, w in (("full script", full), ("10-min path (skip %s)"
                                             % ",".join(map(str, sorted(OPTIONAL))), core)):
        secs = w / a.wpm * 60
        print("%-28s %5d words  %4.1f min at %d wpm  %s"
              % (label, w, secs / 60, a.wpm, "OK" if secs <= a.limit else "OVER"))

    if a.propose:
        print("\nproposed cumulative headers (10-min path, %d wpm):" % a.wpm)
        t = 0.0
        for n, title, _, words in rows:
            if n in OPTIONAL:
                continue
            s, e = t, t + words / a.wpm * 60
            print("  slide %-2d {%d:%02d--%d:%02d}"
                  % (n, int(s // 60), int(s % 60), int(e // 60), int(e % 60)))
            t = e

    sys.exit(0 if core / a.wpm * 60 <= a.limit else 1)


if __name__ == "__main__":
    main()
