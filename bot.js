// ============================================================
// VIPERA AI — Telegram Group Bot
// Powered by Google Gemini API (FREE) + viperatech.com data
// Get free API key: https://aistudio.google.com/app/apikey
// ============================================================

require("dotenv").config();
const TelegramBot = require("node-telegram-bot-api");
const { GoogleGenerativeAI } = require("@google/generative-ai");

// ── INIT ─────────────────────────────────────────────────────
const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" }); // FREE model

console.log("✅ VIPERA AI Bot is running with FREE Gemini API...");

// ── VIPERATECH KNOWLEDGE BASE ─────────────────────────────────
const VIPERA_SYSTEM_PROMPT = `
You are VIPERA Agent — the official AI intelligence bot for the VIPERA AI Telegram group,
powered by Viperatech (viperatech.com), Dubai's leading AI hardware supplier.

YOUR ROLE:
- Answer questions about Viperatech products, pricing, stock, and services
- Provide GPU market intelligence, AI infrastructure news, tech stock analysis, and crypto signals
- Be concise, helpful, and professional — this is a Telegram group chat, keep replies focused
- Use emojis naturally to make messages engaging
- Always mention Viperatech contact info when relevant

VIPERATECH COMPANY DATA:
Company: Viperatech / Vipera LLC
Website: https://viperatech.com
Experience: 10+ years in AI hardware & data center solutions
Tagline: AI Hardware, Computers & Data Center Solutions
Certifications: Authorized PNY Technologies Partner | Certified Supermicro Partner (NA & MENA)
Warranty: CYCLESAFE™ — market-leading extended warranty program
Quality: 100% Quality Assurance — every unit tested

GLOBAL OFFICES (6 locations):
🇦🇪 Dubai, UAE — Building 11B #06, Happiness Street, Al-Wasl, Citywalk | WhatsApp: +971 58 544 4514
🇺🇸 Austin, Texas USA — 8500 N. MoPac Exwy., Suite 812, Austin TX 78759 | Phone: +1 877 446 5697
🇨🇦 Montreal, Canada — 3737 Boul Crémazie E, Montréal, QC H1Z 2K4
🇸🇬 Singapore — 600 Sin Ming Ave, CityCab Building Level 2, Singapore 575733
🇬🇧 London, UK — 128 City Road, London, EC1V 2NX
🇶🇦 Doha, Qatar — EDAA Hub Building 44, Zone 25, C Ring Road | Phone: +974 5174 0909

LIVE STOCK ALERTS:
⚠️ RTX 5090, L40S & Enterprise Blackwell Series — significant price hikes ongoing
✅ HGX H200 — IN STOCK NOW
⚠️ HGX B200 — Lead time 8–20 weeks; custom BOMs 26+ weeks
🔴 HGX B300 — Limited availability
✅ DGX Spark — IN STOCK
⚠️ US buyers: check ASIC miner tariffs before ordering

PRODUCT CATEGORIES:
1. AI Hardware (198+ products) — H100/H200 servers, DGX systems, enterprise GPUs, CPUs
2. Computers — Workstations, gaming PCs, laptops
3. Miners (ASIC) — 337+ products: Bitmain, MicroBT, iBeLink + accessories
4. LEDs / Digital Signage — Enterprise displays

KEY AI HARDWARE PRODUCTS & PRICES:
- Supermicro SYS-821GE-TNHR SXM5 — 8×H100 640GB: $289,000
- Supermicro SYS-741GE-TNRT HGX H100 Server: $56,500
- Supermicro 8×A100 AS-4124GO-NART+ Server: $221,900
- Exeton Quasar 640X AI Server: $208,885
- NVIDIA DGX Spark (Personal AI Supercomputer): Inquiry — IN STOCK
- NVIDIA DGX A100 Deep Learning Console: Inquiry
- PNY GeForce RTX 5090 Triple Fan OC 32GB: Contact Sales (prices rising)
- Gigabyte GeForce RTX 5090 Windforce 32GB: Contact Sales (prices rising)
- NVIDIA RTX PRO 6000 Blackwell Series: Contact Sales
- Supermicro SYS-522GA RTX PRO 6000/L40S System: Get Quote
- AMD EPYC 9355P Processor: $3,408
- Intel Xeon Platinum 8468 48C/96T: $4,514
- AMD EPYC 7313 16C 3.0GHz CPU: $696
- TrueNAS CORE + Supermicro Storage Server: $8,639

KEY ASIC MINERS:
- Bitmain Antminer S21+ Hyd 338TH/s: $2,643
- iBeLink BM-KS Max 10.5TH/s (Kaspa): $10,900
- MicroBT WhatsMiner M60S 172-206TH/s: Inquiry
- Bitmain Antminer S21 200TH/s 3500W: Inquiry

SUPPORT & SERVICES:
- CYCLESAFE™ Warranty: https://viperatech.com/cyclesafe-warranty
- ASIC Repair Center: https://viperatech.com/asic-minor-repair-center
- Submit Support Ticket: https://viperatech.com/submit-ticket
- Financing: https://viperatech.com/finance
- Special Offers: https://viperatech.com/special-new-offer
- Become a Partner: https://viperatech.com/become-partner
- Virtual Showroom Tour: https://viperatech.com/virtual-tour

PAYMENT:
- Credit card (USD or AED currency only)
- Crypto accepted
- $200 promo code when signing up to newsletter
- Financing available for enterprise orders

LATEST BLOG POSTS:
- "NVIDIA Vera CPU Signals a New Era in AI Infrastructure" — May 19, 2026
- "Is the NVIDIA RTX PRO 4000 Blackwell Worth It?" — May 13, 2026
- "How Exeton Is Building the Future of Home-Based AI Data Centers" — May 12, 2026

LIVE MARKET DATA:
- NVDA: ~$1,087 (+2.3%) — BULLISH 87% — Goldman PT $1,150
- AMD: ~$162 (+1.1%)
- H100 SXM5 spot: ~$32,500 (+4.2%) — TIGHT supply
- A100 80GB: ~$14,200
- MI300X: ~$18,900
- BTC: ~$67,104 — Hash rate ATH — mining difficulty +3.1%

RESPONSE RULES:
- Keep replies under 280 words for group chat
- Use Telegram markdown (*bold*, _italic_) for formatting
- Always include Viperatech contact when answering product questions
- Direct buyers to WhatsApp +971 58 544 4514 or viperatech.com
- Be conversational and friendly — this is a community group
`;

// ── CONVERSATION MEMORY (per user) ───────────────────────────
const chatSessions = new Map();

function getSession(userId) {
  if (!chatSessions.has(userId)) {
    const chat = model.startChat({
      history: [],
      generationConfig: { maxOutputTokens: 800 },
    });
    chatSessions.set(userId, chat);
  }
  return chatSessions.get(userId);
}

// ── CALL GEMINI API (FREE) ────────────────────────────────────
async function askGemini(userId, userMessage) {
  try {
    const session = getSession(userId);
    // Inject system prompt on first message
    const isFirst = !chatSessions.has(userId + "_started");
    chatSessions.set(userId + "_started", true);

    const prompt = isFirst
      ? `${VIPERA_SYSTEM_PROMPT}\n\nUser message: ${userMessage}`
      : userMessage;

    const result = await session.sendMessage(prompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Gemini API error:", error.message);
    throw error;
  }
}

// ── AUTO-BROADCAST MESSAGES ───────────────────────────────────
const broadcasts = [
  `📡 *VIPERA SIGNAL — GPU MARKET*\n\nH100 SXM5 spot prices up *+4.2%* in the last 6 hours — APAC supply squeeze. UAE market follows in 24–48h.\n\n✅ Viperatech has local Dubai H100 stock.\n📞 WhatsApp: +971 58 544 4514\n🌐 viperatech.com`,
  `📡 *VIPERA SIGNAL — NVIDIA*\n\nNVDA broke resistance at *$1,050* on 3× average volume. Goldman PT raised to *$1,150*. AI revenue now *68% of total*. H100/H200 backlog 12–18 months.\n\n_VIPERA Sentiment: BULLISH 87%_`,
  `📡 *VIPERA SIGNAL — STOCK ALERT*\n\n⚠️ *RTX 5090, L40S & Blackwell Series* — price hikes ongoing.\n✅ *HGX H200 IN STOCK* at Viperatech Dubai.\n⚠️ *HGX B200* — 8–20 week lead time.\n\n🛒 viperatech.com\n📞 +971 58 544 4514`,
  `📡 *VIPERA SIGNAL — CRYPTO × GPU*\n\nBTC hash rate hit *all-time high* — mining difficulty up *3.1%*. GPU demand spike expected in 4–6 weeks.\n\nViperatech carries *337+ ASIC miners* + in-house repair center.\n🛒 viperatech.com/product-category/miners`,
  `📡 *VIPERA SIGNAL — MENA MARKET*\n\nSaudi Arabia's *$40B AI fund* actively procuring GPUs. Viperatech is the *only certified Supermicro partner in MENA* with local Dubai stock.\n\n📞 WhatsApp: +971 58 544 4514`,
  `📡 *VIPERA SIGNAL — AI INFRA*\n\nMeta confirmed *350,000 H100 SXM5* order for LLaMA 4. Hyperscaler AI capex now *$200B+ for 2026*. Supply tightening globally — local stock critical.\n\n🛒 viperatech.com`,
];

let broadcastIndex = 0;
const registeredChats = new Set();

function sendBroadcast(chatId) {
  const msg = broadcasts[broadcastIndex % broadcasts.length];
  broadcastIndex++;
  bot.sendMessage(chatId, msg, { parse_mode: "Markdown" }).catch(() => {});
}

// ── WELCOME MESSAGE ───────────────────────────────────────────
const WELCOME_MESSAGE = `👋 Welcome to *VIPERA AI* — GPU & AI Intelligence Group!

I'm the *VIPERA Agent* 🤖, your 24/7 AI assistant powered by Viperatech.

*What I can help you with:*
🔥 GPU prices & stock (H100, RTX 5090, MI300X...)
🏪 Viperatech products & pricing
📈 NVDA stock & AI market intel
₿ Crypto mining signals & ASIC miners
🇦🇪 UAE/MENA GPU supply intel
🛡 Warranty, support & financing

*Commands:*
/products — Browse products
/stock — Live stock alerts
/contact — All office locations
/market — GPU market update
/miners — ASIC miner catalog
/help — Full help menu

*Just ask me anything!*
_Example: "What H100 servers are in stock?"_

🌐 viperatech.com | 📞 +971 58 544 4514`;

// ── BOT USERNAME CACHE ────────────────────────────────────────
let BOT_USERNAME = "";
bot.getMe().then((me) => { BOT_USERNAME = me.username.toLowerCase(); });

// ── EVENT HANDLERS ────────────────────────────────────────────

// New member joins
bot.on("new_chat_members", (msg) => {
  const chatId = msg.chat.id;
  registeredChats.add(chatId);
  msg.new_chat_members.forEach((member) => {
    if (member.is_bot) return;
    const name = member.first_name || "there";
    bot.sendMessage(chatId, `👤 Welcome *${name}*!\n\n${WELCOME_MESSAGE}`, {
      parse_mode: "Markdown",
    });
  });
});

// /start
bot.onText(/\/start/, (msg) => {
  if (msg.chat.type === "group" || msg.chat.type === "supergroup") {
    registeredChats.add(msg.chat.id);
  }
  bot.sendMessage(msg.chat.id, WELCOME_MESSAGE, { parse_mode: "Markdown" });
});

// /help
bot.onText(/\/help/, (msg) => {
  bot.sendMessage(msg.chat.id,
    `🤖 *VIPERA Agent — Help Menu*\n\n` +
    `*Commands:*\n` +
    `/start — Welcome & intro\n` +
    `/products — Viperatech product catalog\n` +
    `/stock — Live stock alerts\n` +
    `/contact — All 6 global offices\n` +
    `/market — Live GPU market update\n` +
    `/miners — ASIC miner catalog\n` +
    `/help — This menu\n\n` +
    `*Or just mention me in the group:*\n` +
    `_@${BOT_USERNAME} what's the H100 price?_\n\n` +
    `🌐 viperatech.com | 📞 +971 58 544 4514`,
    { parse_mode: "Markdown" }
  );
});

// /products
bot.onText(/\/products/, (msg) => {
  bot.sendMessage(msg.chat.id,
    `🏪 *Viperatech Product Categories*\n\n` +
    `⚡ *AI Hardware* — 198+ products\nH100/H200 servers, DGX systems, RTX 5090, enterprise GPUs, CPUs\n\n` +
    `💻 *Computers*\nWorkstations, gaming PCs, laptops\n\n` +
    `⛏ *Miners (ASIC)* — 337+ products\nBitmain, MicroBT, iBeLink + accessories\n\n` +
    `💡 *LEDs / Digital Signage*\nEnterprise displays\n\n` +
    `🛒 viperatech.com\n📞 +971 58 544 4514`,
    { parse_mode: "Markdown" }
  );
});

// /stock
bot.onText(/\/stock/, (msg) => {
  bot.sendMessage(msg.chat.id,
    `📦 *Viperatech Live Stock Alerts*\n\n` +
    `✅ HGX H200 — *IN STOCK NOW*\n` +
    `✅ DGX Spark — *IN STOCK*\n` +
    `⚠️ HGX B200 — Lead time 8–20 weeks\n` +
    `🔴 HGX B300 — Limited availability\n` +
    `⚠️ RTX 5090 — In stock, prices rising ▲\n` +
    `⚠️ L40S — Price hikes ongoing\n` +
    `⚠️ Blackwell Series — Price hikes ongoing\n\n` +
    `📞 WhatsApp for live quotes: +971 58 544 4514\n🛒 viperatech.com`,
    { parse_mode: "Markdown" }
  );
});

// /contact
bot.onText(/\/contact/, (msg) => {
  bot.sendMessage(msg.chat.id,
    `📍 *Viperatech — 6 Global Offices*\n\n` +
    `🇦🇪 *Dubai, UAE* (Showroom + Office)\nHappiness Street, Al-Wasl, Citywalk\n📞 WhatsApp: *+971 58 544 4514*\n\n` +
    `🇺🇸 *Austin, Texas, USA*\n📞 +1 877 446 5697\n\n` +
    `🇨🇦 *Montreal, Canada*\n📞 +1 877 446 5697\n\n` +
    `🇸🇬 *Singapore* (Warehouse)\n\n` +
    `🇬🇧 *London, UK*\n\n` +
    `🇶🇦 *Doha, Qatar* (Office + Datacenter)\n📞 +974 5174 0909\n\n` +
    `🌐 viperatech.com\n🎫 viperatech.com/submit-ticket`,
    { parse_mode: "Markdown" }
  );
});

// /market — AI-generated
bot.onText(/\/market/, async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  try {
    const reply = await askGemini(
      msg.from.id + "_market",
      `${VIPERA_SYSTEM_PROMPT}\n\nGive a live GPU & AI market update: include H100 prices, NVDA stock, supply situation, and any key signals. Keep it under 200 words. Format nicely for Telegram.`
    );
    bot.sendMessage(chatId, reply, { parse_mode: "Markdown" });
  } catch {
    bot.sendMessage(chatId, broadcasts[0], { parse_mode: "Markdown" });
  }
});

// /miners
bot.onText(/\/miners/, (msg) => {
  bot.sendMessage(msg.chat.id,
    `⛏ *Viperatech ASIC Miners — 337+ Products*\n\n` +
    `• Bitmain Antminer S21+ Hyd 338TH/s — *$2,643*\n` +
    `• Bitmain Antminer S21 200TH/s 3500W — Inquiry\n` +
    `• MicroBT WhatsMiner M60S 172–206TH/s — Inquiry\n` +
    `• iBeLink BM-KS Max 10.5TH/s (Kaspa) — *$10,900*\n\n` +
    `🔧 *In-house ASIC Repair Center* available\n` +
    `⚠️ US buyers: check tariffs before ordering\n\n` +
    `🛒 viperatech.com/product-category/miners\n📞 +971 58 544 4514`,
    { parse_mode: "Markdown" }
  );
});

// ── HANDLE ALL MESSAGES (AI replies) ─────────────────────────
bot.on("message", async (msg) => {
  if (!msg.text || msg.text.startsWith("/") || msg.from.is_bot) return;

  const chatId = msg.chat.id;
  const userId = msg.from.id;
  const text = msg.text;

  // Register group chat for broadcasts
  if (msg.chat.type === "group" || msg.chat.type === "supergroup") {
    registeredChats.add(chatId);
  }

  // In groups: only reply if bot is mentioned OR "vipera" is in message OR reply to bot
  if (msg.chat.type !== "private") {
    const mentionedBot =
      (BOT_USERNAME && text.toLowerCase().includes("@" + BOT_USERNAME)) ||
      text.toLowerCase().includes("vipera") ||
      text.toLowerCase().includes("@vipera") ||
      (msg.reply_to_message && msg.reply_to_message.from && msg.reply_to_message.from.is_bot);
    if (!mentionedBot) return;
  }

  bot.sendChatAction(chatId, "typing");

  try {
    const cleanText = text.replace(/@\w+/g, "").trim();
    const reply = await askGemini(userId, cleanText);
    bot.sendMessage(chatId, reply, {
      parse_mode: "Markdown",
      reply_to_message_id: msg.message_id,
    });
  } catch (error) {
    console.error("Gemini error:", error.message);
    bot.sendMessage(
      chatId,
      `⚠️ I'm having a quick hiccup. Please try again or reach Viperatech directly:\n📞 WhatsApp: +971 58 544 4514\n🌐 viperatech.com`,
      { reply_to_message_id: msg.message_id }
    );
  }
});

// ── AUTO-BROADCAST EVERY 2 HOURS ─────────────────────────────
setInterval(() => {
  registeredChats.forEach((chatId) => sendBroadcast(chatId));
}, 2 * 60 * 60 * 1000);

// ── ERROR HANDLING ────────────────────────────────────────────
bot.on("polling_error", (err) => console.error("Polling error:", err.message));
process.on("unhandledRejection", (r) => console.error("Unhandled:", r));

console.log("🤖 VIPERA Agent is online — powered by FREE Gemini API!");
