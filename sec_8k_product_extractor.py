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
    """HTTP GET with the SEC-required User-Agent and a small delay to respect the rate limit."""
    time.sleep(0.15)  # SEC fair access: <= 10 requests/second
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


def sp500_tickers():
    """Return the current S&P 500 tickers (Wikipedia table), or a fallback list if the page is unreachable."""
    try:
        html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
        table = BeautifulSoup(html, "lxml").find("table", id="constituents")
        return [row.find("td").get_text(strip=True).replace(".", "-") for row in table.find_all("tr")[1:]]
    except Exception as e:
        print(f"[warn] could not load S&P 500 list ({e}); using fallback list")
        return FALLBACK_TICKERS


def ticker_map():
    """Map ticker -> (10-digit CIK, company title) using the SEC company_tickers.json file."""
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"]: (str(v["cik_str"]).zfill(10), v["title"]) for v in data.values()}


def recent_8ks(cik, limit, min_date="2025-01-01"):
    """Latest 8-Ks that carry an announcement item (7.01 Reg FD / 8.01 Other events)."""
    d = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    r = d["filings"]["recent"]
    out = []
    for form, acc, doc, when, items in zip(r["form"], r["accessionNumber"], r["primaryDocument"],
                                           r["acceptanceDateTime"], r["items"]):
        if when[:10] < min_date:  # submissions are newest-first, so everything after this is older too
            break
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
companies; executives; or events; infrastructure, construction, capacity or capital-expenditure plans; IPOs, joint
ventures, contracts or awards; internal programs, initiatives or restructurings. A product must be something a
customer can buy or use (hardware, software, drug, consumer good, service). For pharma/biotech, include a drug only if it is newly launched, newly approved or
newly unveiled in this filing - NOT a clinical-trial data update or regulatory acceptance for an already known drug.
Most filings have 0 or 1 new product; there may be more.
NER hints (may be noisy) - organizations: {orgs}; product-like entities: {prods}.
For each product give: "name" (the product name only) and "description" (one concise sentence, UNDER 170 characters).
Return ONLY JSON: {{"products": [{{"name": str, "description": str}}]}}. If none: {{"products": []}}.

Filing excerpt:
{text}"""

BAD = re.compile(r"\b(notes?|bonds?|debentures?|warrants?|credit facility|loan|stock|preferred|shares?|presentation|options|conference)\b", re.I)


def clip(s, n=MAX_DESC):
    """Collapse whitespace and cut the text at a word boundary so it stays under the 180-character limit."""
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def extract_products(company, ticker, text, orgs, prods):
    """Ask the LLM for NEW products in the excerpt; returns a list of (name, description) tuples."""
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


JUDGE = """You are a strict fact-checker. Below are candidate "new products" extracted from an SEC 8-K of {company}.
Keep ONLY items that are a genuinely NEW commercial product, service, platform or feature that the company launched
or unveiled for customers. Reject: clinical-trial results, regulatory filings/acceptances/approvals of already-known
drugs, financings, IPOs, joint ventures, contracts, infrastructure/energy/capacity projects, internal programs or
initiatives, restructurings, earnings items, and products that merely resumed shipping or changed pharmacy.
Return ONLY JSON: {{"keep": [indices of items to keep]}}.

{items}"""


def verify_products(company, found):
    """Second LLM pass (precision filter): keep only candidates that are truly new product launches."""
    if not found:
        return found
    from openai import OpenAI
    listing = "\n".join(f"{i}. {n}: {d}" for i, (n, d) in enumerate(found))
    resp = OpenAI().chat.completions.create(
        model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE.format(company=company, items=listing)}],
    )
    try:
        keep = {int(i) for i in json.loads(resp.choices[0].message.content).get("keep", [])}
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return found  # if the judge fails, fall back to the first-pass result
    return [x for i, x in enumerate(found) if i in keep]


FIELDS = ["company_name", "stock_name", "filing_time", "new_product", "product_description"]


def dedupe(rows):
    """Drop repeated (ticker, product, filing date) rows that appear when a filing is mentioned twice."""
    seen, out = set(), []
    for r in rows:
        key = (r["stock_name"], r["new_product"].lower(), r["filing_time"][:10])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_csv(rows, path):
    """Write the extracted rows to CSV using the exact column names required by the assignment."""
    rows = dedupe(rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def main():
    """Walk through S&P 500 companies, analyse their 8-Ks and stop once enough product rows are collected."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="comma-separated; default = S&P 500")
    ap.add_argument("--target-rows", type=int, default=100, help="stop once the CSV has this many product rows")
    ap.add_argument("--max-docs", type=int, default=2000, help="safety cap on the number of filings analysed")
    ap.add_argument("--per-company", type=int, default=5, help="max 8-Ks (Item 7.01/8.01) read per company")
    ap.add_argument("--max-rows-per-company", type=int, default=4, help="cap rows per company so the sample is diverse")
    ap.add_argument("--min-date", default="2025-01-01", help="ignore 8-Ks filed before this date (YYYY-MM-DD)")
    ap.add_argument("--out", default="products.csv")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed so the sample spans the whole index")
    args = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")  # NER model: ORG / PRODUCT entities
    tmap = ticker_map()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else sp500_tickers()
    if not args.tickers:
        random.Random(args.seed).shuffle(tickers)  # avoid an alphabetical (A-C only) sample

    rows, docs, companies = [], 0, set()
    for t in tickers:
        if len(rows) >= args.target_rows or docs >= args.max_docs:
            break
        if t not in tmap:
            continue
        cik, _title = tmap[t]
        try:
            filings = recent_8ks(cik, args.per_company, args.min_date)
        except Exception as e:
            print(f"[error] {t}: {e}")
            continue
        company_rows = 0
        for f in filings:
            if len(rows) >= args.target_rows or docs >= args.max_docs or company_rows >= args.max_rows_per_company:
                break
            try:
                text = filing_text(cik, f)                                   # main doc + EX-99 exhibits
                excerpt, orgs, prods = ner_context(nlp, text)                # NER filtering
                found = extract_products(f["name"], t, excerpt, orgs, prods) if excerpt else []  # LLM pass 1
                found = verify_products(f["name"], found)[: args.max_rows_per_company - company_rows]  # LLM pass 2
            except Exception as e:
                if "401" in str(e):
                    sys.exit("OpenAI rejected the API key (401). Fix OPENAI_API_KEY in .env and retry.")
                print(f"[error] {t} {f['acc']}: {e}", flush=True)
                continue
            docs += 1
            companies.add(t)
            company_rows += len(found)
            for name, desc in found:
                rows.append({"company_name": f["name"], "stock_name": t, "filing_time": f["time"],
                             "new_product": name, "product_description": desc})
            print(f"[docs={docs} rows={len(rows)}/{args.target_rows}] {t} {f['time'][:10]}: "
                  f"{len(found)} new product(s)", flush=True)
        write_csv(rows, args.out)  # save progress after every company

    write_csv(rows, args.out)
    print(f"DONE: {docs} filings from {len(companies)} companies -> {len(rows)} rows in {args.out}")


if __name__ == "__main__":
    main()
