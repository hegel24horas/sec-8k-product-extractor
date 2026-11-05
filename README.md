# SEC 8-K New Product Extractor

Assignment 2 - LLM Document Analysis. Extracts newly announced products from SEC 8-K filings
(S&P 500) using spaCy NER + an OpenAI model and writes `products.csv`
(`company_name, stock_name, filing_time, new_product, product_description`).

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # then fill in your values
python sec_8k_product_extractor.py            # 100 filings
python sec_8k_product_extractor.py --max-docs 20 --tickers AAPL,MSFT
```
See `report.docx` for the methodology.
