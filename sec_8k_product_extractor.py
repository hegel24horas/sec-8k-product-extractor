"""Assignment 2 - LLM Document Analysis.

Extract NEW PRODUCT announcements from SEC 8-K filings (S&P 500 companies) and save them as CSV:
    company_name, stock_name, filing_time, new_product, product_description

Pipeline
  1. Ticker -> CIK         https://www.sec.gov/files/company_tickers.json
  2. CIK -> recent 8-Ks    https://data.sec.gov/submissions/CIK##########.json
                           (only Items 7.01 / 8.01 = press-release style announcements)
  3. Filing -> text        main document + EX-99 exhibits (press releases), parsed with BeautifulSoup
  4. NER (spaCy)           keep sentences with PRODUCT entities or launch language; ORG/PRODUCT entities become hints
  5. LLM (OpenAI)          structured JSON: new products + <180-char description
  6. CSV

Config (env or .env file next to this script):
  SEC_USER_AGENT   "Name email"  (required by the SEC)
  OPENAI_API_KEY   your key
  OPENAI_MODEL     default gpt-4o-mini

Usage:
  python sec_8k_product_extractor.py                  # 100 docs from the S&P 500
  python sec_8k_product_extractor.py --max-docs 20 --per-company 1 --tickers AAPL,MSFT
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def _load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


_load_env()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
UA = os.environ.get("SEC_USER_AGENT", "")
if not UA or "dominio.com" in UA:
    sys.exit("Set SEC_USER_AGENT to a real 'Name email' string (the SEC requires it).")
_k = os.environ.get("OPENAI_API_KEY", "")
if not _k.startswith("sk-") or len(_k) < 40:
    sys.exit("OPENAI_API_KEY missing or not a real key. Put it in .env (OPENAI_API_KEY=sk-...).")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

FALLBACK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "CSCO", "ORCL", "CRM",
                    "ADBE", "IBM", "QCOM", "AVGO", "NFLX", "DIS", "NKE", "KO", "PEP", "PFE", "MRK", "JNJ", "ABT",
                    "GM", "F", "HD", "LOW", "COST"]
ITEMS_OK = ("7.01", "8.01")
MAX_DESC = 179  # "less than 180 characters"


def get(url, **kw):
    time.sleep(0.15)  # SEC fair access: <= 10 requests/second
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


def sp500_tickers():
    try:
        html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
        table = BeautifulSoup(html, "lxml").find("table", id="constituents")
        return [row.find("td").get_text(strip=True).replace(".", "-") for row in table.find_all("tr")[1:]]
    except Exception as e:
        print(f"[warn] could not load S&P 500 list ({e}); using fallback list")
        return FALLBACK_TICKERS


def ticker_map():
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"]: (str(v["cik_str"]).zfill(10), v["title"]) for v in data.values()}


def recent_8ks(cik, limit):
    """Latest 8-Ks that carry an announcement item (7.01 Reg FD / 8.01 Other events)."""
    d = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    r = d["filings"]["recent"]
    out = []
    for form, acc, doc, when, items in zip(r["form"], r["accessionNumber"], r["primaryDocument"],
                                           r["acceptanceDateTime"], r["items"]):
        if form == "8-K" and any(i in items.split(",") for i in ITEMS_OK):
            out.append({"acc": acc, "doc": doc, "time": when, "name": d["name"]})
            if len(out) >= limit:
                break
    return out


def filing_text(cik, f):
    """Main 8-K document plus EX-99 exhibits (where press releases live)."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['acc'].replace('-', '')}/"
    urls = [base + f["doc"].split("/")[-1]]
    try:
        idx = BeautifulSoup(get(base + f["acc"] + "-index.htm").text, "lxml")
        for tr in idx.select("table.tableFile tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) >= 4 and tds[3].get_text(strip=True).startswith("EX-99"):
                a = tds[2].find("a")
                if a and a["href"].lower().endswith((".htm", ".html")):
                    urls.append("https://www.sec.gov" + a["href"].replace("/ix?doc=", ""))
    except Exception:
        pass
    parts = []
    for u in dict.fromkeys(urls):
        try:
            parts.append(" ".join(BeautifulSoup(get(u).text, "lxml").get_text(" ").split()))
        except Exception:
            continue
    return " ".join(parts)


LAUNCH = ("launch", "introduc", "unveil", "announce", "releas", "new ", "available", "debut", "rolls out", "expands")


def ner_context(nlp, text, max_chars=7000):
    """NER step: keep sentences with PRODUCT entities or launch language; return (excerpt, hints)."""
    keep, hints = [], {"ORG": [], "PRODUCT": []}
    for sent in nlp(text[:250000]).sents:
        s = sent.text.strip()
        prod = [e.text for e in sent.ents if e.label_ == "PRODUCT"]
        if len(s) < 30 or not (prod or any(k in s.lower() for k in LAUNCH)):
            continue
        keep.append(s)
        hints["PRODUCT"] += prod
        hints["ORG"] += [e.text for e in sent.ents if e.label_ == "ORG"]
        if sum(map(len, keep)) > max_chars:
            break
    uniq = lambda xs: list(dict.fromkeys(xs))[:15]
    return " ".join(keep)[:max_chars], uniq(hints["ORG"]), uniq(hints["PRODUCT"])


PROMPT = """You analyze an SEC 8-K filing from {company} ({ticker}).
Extract ONLY NEW products, services or platforms that the company is announcing or launching in this filing.
Do NOT include: existing/legacy products or business segments; financial instruments (notes, bonds, stock, preferred
shares, options, credit facilities); earnings results; presentations, reports or conferences; acquisitions of
companies; executives; or events. For pharma/biotech, include a drug only if it is newly launched, newly approved or
newly unveiled in this filing - NOT a clinical-trial data update or regulatory acceptance for an already known drug.
Most filings have 0 or 1 new product; there may be more.
NER hints (may be noisy) - organizations: {orgs}; product-like entities: {prods}.
For each product give: "name" (the product name only) and "description" (one concise sentence, UNDER 170 characters).
Return ONLY JSON: {{"products": [{{"name": str, "description": str}}]}}. If none: {{"products": []}}.

Filing excerpt:
{text}"""

BAD = re.compile(r"\b(notes?|bonds?|debentures?|warrants?|credit facility|loan|stock|preferred|shares?|presentation|options|conference)\b", re.I)


def clip(s, n=MAX_DESC):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def extract_products(company, ticker, text, orgs, prods):
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(
            company=company, ticker=ticker, orgs=", ".join(orgs) or "none", prods=", ".join(prods) or "none", text=text)}],
    )
    try:
        items = json.loads(resp.choices[0].message.content).get("products", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    for p in items:
        name, desc = str(p.get("name", "")).strip(), str(p.get("description", "")).strip()
        if name and desc and not BAD.search(name + " " + desc):
            out.append((name, clip(desc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="comma-separated; default = S&P 500")
    ap.add_argument("--max-docs", type=int, default=100, help="stop after this many filings")
    ap.add_argument("--per-company", type=int, default=2)
    ap.add_argument("--out", default="products.csv")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed so the sample spans the whole index")
    args = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")
    tmap = ticker_map()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else sp500_tickers()

    if not args.tickers:
        random.Random(args.seed).shuffle(tickers)
    rows, docs = [], 0
    for t in tickers:
        if docs >= args.max_docs:
            break
        if t not in tmap:
            continue
        cik, title = tmap[t]
        try:
            filings = recent_8ks(cik, args.per_company)
        except Exception as e:
            print(f"[error] {t}: {e}")
            continue
        for f in filings:
            if docs >= args.max_docs:
                break
            try:
                text = filing_text(cik, f)
                excerpt, orgs, prods = ner_context(nlp, text)
                found = extract_products(f["name"], t, excerpt, orgs, prods) if excerpt else []
            except Exception as e:
                if "401" in str(e):
                    sys.exit("OpenAI rejected the API key (401). Fix OPENAI_API_KEY in .env and retry.")
                print(f"[error] {t} {f['acc']}: {e}")
                continue
            docs += 1
            print(f"[{docs}/{args.max_docs}] {t} {f['time'][:10]}: {len(found)} new product(s)")
            for name, desc in found:
                rows.append({"company_name": f["name"], "stock_name": t, "filing_time": f["time"],
                             "new_product": name, "product_description": desc})

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["company_name", "stock_name", "filing_time", "new_product",
                                           "product_description"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"Processed {docs} filings, wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
