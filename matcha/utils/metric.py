import re, unicodedata

# 영어용 간단 정규화 (소문자, 기호 제거, 공백 정리)
_punct = re.compile(r"[^\w\s]")
def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _punct.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lev(a, b):  # Levenshtein distance
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def wer_en(ref: str, hyp: str) -> float:
    r = _normalize(ref).split()
    h = _normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _lev(r, h) / len(r)

def cer_en(ref: str, hyp: str) -> float:
    r = list(_normalize(ref).replace(" ", ""))
    h = list(_normalize(hyp).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _lev(r, h) / len(r)