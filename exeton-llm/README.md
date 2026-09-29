# Exeton Sales-Engineer Chatbot on NVIDIA DGX Spark

A chatbot that answers like an Exeton sales engineer. It gives product details, specs, **price** and **availability**, all taken from https://exeton.com, and runs fully on-prem on a DGX Spark.

---

## 1. Should you train an LLM "from scratch"? Short answer: no

| Option | What it means | Verdict for this project |
|---|---|---|
| **Pre-train from scratch** | Random weights, learn language + knowledge from raw text | ❌ Not viable (see below) |
| **Fine-tune (LoRA) an open model** | Start from Llama / Qwen / Mistral / Nemotron, teach it Exeton's *style & behaviour* | ✅ Optional, step 5 |
| **RAG (retrieval-augmented generation)** | Scrape the site → search it on every question → LLM answers from the retrieved facts | ✅ **Required.** This is where price/stock come from |

Why pre-training from scratch doesn't fit:

1. **Not enough data.** A model needs hundreds of billions to trillions of tokens to learn English and reasoning. All of exeton.com is maybe a few million tokens, which is roughly 100,000× too little. A model trained only on that would write broken sentences.
2. **Not enough compute.** Even a small 1B-parameter model trained on 20B tokens takes about 6 × 10⁹ × 2 × 10¹⁰ ≈ 1.2 × 10²⁰ FLOPs. DGX Spark peaks at about 1 PFLOP (FP4, sparse). At realistic BF16 training throughput that is weeks of runtime, and the result would still be far worse than a free open model.
3. **Prices and stock change daily.** Anything baked into model weights is stale the next day. Facts that change must be **looked up at answer time** (RAG + live check), not memorised.

So the right design on DGX Spark is:

```
            nightly                      every question
exeton.com ─────────► scrape ─► index      user ─► retrieve top products/pages
                                             │        │
                                             │        ├─► live price/stock check (exeton.com API)
                                             │        ▼
                                             └── open LLM (optionally LoRA-tuned) on DGX Spark
                                                      │
                                                      ▼
                                   answer: specs + price + availability + link
```

---

## 2. What's in this folder

| File | Step | Purpose |
|---|---|---|
| `probe_stack.py` | 1 | Shows how the site is built (framework, server-rendered vs JavaScript, product APIs) |
| `scrape_exeton.py` | 1 | Scrapes the whole site: `/all-products` + sitemap → every `/product-details/` page (JSON-LD or visible page). Writes `data/products.jsonl`, `data/pages.jsonl` |
| `build_index.py` | 2 | Embeds products/pages on the GPU → `data/index/` |
| `chat_server.py` | 3 | FastAPI `/chat` endpoint: retrieval + live price/stock refresh + LLM call |
| `make_sft_dataset.py` | 4 | (optional) Builds a fine-tuning dataset from the catalog |
| `train_lora.py` | 5 | (optional) LoRA fine-tunes an open model on DGX Spark and merges it |
| `refresh.sh` | – | Nightly re-scrape + re-index + restart (cron) |
| `deploy/exeton-chat.service` | – | systemd service that keeps the chatbot running |
| `static/chat.html` | – | Browser chat page served at `/` |
| `knowledge.py` | – | Shared formatting + the sales-engineer system prompt |

---

## 3. DGX Spark setup

DGX Spark facts that matter here: GB10 Grace Blackwell superchip, **ARM64 (aarch64)** CPU, **128 GB unified CPU/GPU memory**, DGX OS (Ubuntu-based) with CUDA and Docker + NVIDIA Container Toolkit preinstalled. Use **NGC containers**. They are built for ARM64 + Blackwell, so you avoid pip wheel problems.

```bash
# On the Spark
git clone <this repo> && cd vipera-ai-bot/exeton-llm
export HF_TOKEN=hf_xxx   # needed for gated models such as Llama

# PyTorch container (for index building + fine-tuning)
docker run --gpus all -it --rm --ipc=host \
  -v $PWD:/work -w /work -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN nvcr.io/nvidia/pytorch:<latest>-py3 bash
pip install -r requirements.txt
```

Pick the current `pytorch` tag from https://catalog.ngc.nvidia.com. NVIDIA also publishes DGX Spark "playbooks" (vLLM, Ollama, fine-tuning, NIM) at https://build.nvidia.com/spark.

---

## 4. Step by step

### Step 1: check how the site is built, then scrape exeton.com
exeton.com is **not WooCommerce**. Products live at `/product-details/<slug>` and are listed on `/all-products`. First run the probe on the Spark (or any machine with internet):
```bash
python probe_stack.py https://exeton.com/product-details/h274-a81
python probe_stack.py https://exeton.com/product-details/h274-a81 --browser   # + list the page's API/JSON calls
```
It reports the framework (Next.js, Laravel, …), whether price and stock are already in the raw HTML or only appear after JavaScript runs, any structured data (JSON-LD, `__NEXT_DATA__`), and, with `--browser`, the JSON API calls that return product data.

Then scrape:
```bash
python scrape_exeton.py                  # full site
python scrape_exeton.py --max-pages 30   # quick test
python scrape_exeton.py --render         # only if the probe says content is rendered by JavaScript
```
- Product URLs come from `/all-products` (with pagination) plus the sitemap, if there is one. Menu and footer links (about, contact, warranty…) are saved as company pages.
- Each product page is read from schema.org JSON-LD if present. Otherwise it's read from the visible page: `<h1>` title, first price after the title (`$`, `USD`, `AED`, `€`, `£`), stock wording ("In stock", "Out of stock", "Lead time", "Request a quote"…), spec tables, breadcrumbs. The full page text is kept, so the LLM still sees every detail.
- WooCommerce's Store API is tried first and simply skipped on exeton.com. The live price/stock re-check in `chat_server.py` only works with that API, so here answers use the last scrape: **run the nightly refresh** (step 6), or every few hours.
- It respects `robots.txt` and waits between requests (`--delay`).
- Check the output: `head -n 3 data/products.jsonl`. Compare a few products with the website. If a field is wrong, adjust `parse_visible_product()`, or send the probe output so it can be matched to the site's exact HTML or API.

### Step 2: build the search index
```bash
python build_index.py     # default embedder BAAI/bge-base-en-v1.5 (runs on the GPU)
```

### Step 3: serve an LLM on the Spark
Any OpenAI-compatible server works. Start with a strong instruct model (no training needed yet):

```bash
# Option A: vLLM (NGC vLLM container), best throughput
docker run --gpus all --rm -p 8001:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN \
  nvcr.io/nvidia/vllm:<latest>-py3 \
  vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name exeton-sales --max-model-len 16384

# Option B: Ollama, easiest
ollama pull llama3.1:8b && ollama serve   # then LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1:8b
```
128 GB of unified memory can hold a 70B model in FP8/4-bit, or around 120B in FP4, for higher-quality answers. An 8B model gives the fastest responses.

### Step 3b: start the chatbot API
```bash
LLM_BASE_URL=http://localhost:8001/v1 LLM_MODEL=exeton-sales \
  uvicorn chat_server:app --host 0.0.0.0 --port 8000

curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Do you have H200 servers in stock and what is the price?"}' | jq
```
The response contains `reply` plus `sources` (product URLs used). With `LIVE_REFRESH=1` (default), each retrieved product's price and stock is re-checked on exeton.com at question time (5-minute cache). Customers therefore get current availability even between nightly scrapes.

**At this point the chatbot is complete and useful.** Steps 4–5 are optional polish.

### Step 4 (optional): build a fine-tuning dataset
```bash
python make_sft_dataset.py                          # template answers
# or, better, let a big model on the Spark write the answers ("teacher"):
python make_sft_dataset.py --teacher-url http://localhost:8001/v1 --teacher-model exeton-sales
```
Every example includes the same CONTEXT block the server uses. The model learns to **read context and answer like a sales engineer**: quote exact price/stock/URL, ask qualifying questions (workload, model size, budget, power), and say "I don't have that" instead of making things up. It does **not** memorise prices.

Add your own real data too, because it has the biggest impact: past sales chats and emails (anonymised), FAQ, datasheets, quotes templates. Aim for 1–5k good examples and **review a sample by hand**.

### Step 5 (optional): LoRA fine-tune on DGX Spark
```bash
python train_lora.py --base meta-llama/Llama-3.1-8B-Instruct --epochs 2
# → outputs/exeton-merged/   then serve it:
vllm serve /work/outputs/exeton-merged --served-model-name exeton-sales
```
- An 8B model in BF16 with LoRA fits easily in 128 GB. A few thousand examples take on the order of tens of minutes to a few hours.
- 70B-class bases need QLoRA (4-bit) and a much longer run. Try 8B first and compare answers.
- Evaluate before switching: ask 50 real customer questions to the base model and to the tuned model (both with RAG), and check the prices, stock and links against the website.

### Step 6: keep it running + keep it fresh
Run the chatbot as a background service. It starts on boot and restarts if it crashes. It uses port **8010**, because 8000 is taken by the Vipera AI API container on this Spark.
```bash
cd ~/vipera-ai-bot/exeton-llm
mkdir -p ~/.config/systemd/user
cp deploy/exeton-chat.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now exeton-chat.service
sudo loginctl enable-linger $USER          # keep it running when you log out / after reboot

systemctl --user status exeton-chat.service   # should say "active (running)"
journalctl --user -u exeton-chat.service -f   # live logs (Ctrl+C to exit)
```
Chat in the browser at **http://<spark-ip>:8010/**. Find the IP with `hostname -I`.

Nightly re-scrape and re-index, which restarts the service with fresh prices and stock:
```bash
crontab -e
# add this line (runs 02:00 every night):
0 2 * * * $HOME/vipera-ai-bot/exeton-llm/refresh.sh >> $HOME/exeton-refresh.log 2>&1
```
If the site is down or returns far fewer products than last time, the scrape **keeps the old data** and the log explains why. When the drop is real, run `python scrape_exeton.py --force`. Re-scraping is all that's needed when products or prices change. **No retraining.**

To change the model or port, edit `~/.config/systemd/user/exeton-chat.service`, then run `systemctl --user daemon-reload && systemctl --user restart exeton-chat.service`.

---

## 5. Connecting it to a chat front-end (Telegram / website)

The API is plain HTTP, so any front-end can call it. For example, from the Node bot in this repo:

```js
const res = await fetch("http://<spark-ip>:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: userText, history }), // history: [{role, content}, ...]
});
const { reply, sources } = await res.json();
```

---

## 6. Checklist / tips

- **Accuracy first**: `temperature 0.2` and "answer only from CONTEXT" rules are already set. Always show the product URL so customers can verify.
- **Model numbers matter**: retrieval boosts exact matches on tokens like `H200`, `B200`, `RTX 5090`, `SYS-821GE`.
- **Prices shown "on request"** are passed through as *"Price on request (contact sales for a quote)"*.
- **Security**: put the API behind your reverse proxy/auth. Don't expose the vLLM port publicly.
- **Legal**: you're scraping your own company's site. Still keep the crawl polite (`--delay`) so it doesn't load the production shop.
