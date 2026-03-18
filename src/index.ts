import "dotenv/config";
import express, { Request, Response } from "express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

const app = express();
app.use(express.json({ limit: "1mb" }));

// ─── LLM ─────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0.1,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// ─── Prompt ──────────────────────────────────────────────────
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an expert AI playing the board game Splendor. Your goal is to reach 15 prestige points before your opponent.

=== GAME STATE STRUCTURE ===
- info: game metadata (gameId, state, currentTurn, players list)
- players: dict playerId -> {{ gems, bonuses, reservedCards, purchasedCards, points }}
  - gems: current gems in hand e.g. {{ "White": 2, "Blue": 1, ... }}
  - bonuses: permanent discounts from purchased cards e.g. {{ "White": 1, ... }}
  - reservedCards: array of cardId (max 3)
  - purchasedCards: array of cardId
  - points: current prestige points
- board.gemBank: gems remaining on the board e.g. {{ "White": 4, "Gold": 5, ... }}
- board.visibleCards: {{ level1, level2, level3 }} each is array of {{ cardId, level, points, bonusColor, cost }}
- board.nobles: array of {{ nobleId, points, requirements }} where requirements is dict of color -> count needed in bonuses
- turn: {{ currentPlayer, currentPlayerIndex, phase, turnNumber }}

=== CRITICAL: AFFORD CHECK (verify EVERY time before PURCHASE_CARD) ===
Step 1: For each color, effectiveCost[color] = max(0, card.cost[color] - player.bonuses[color])
Step 2: For each color, shortage[color] = max(0, effectiveCost[color] - player.gems[color])
Step 3: goldNeeded = sum of all shortage values
Step 4: You can afford ONLY IF player.gems[Gold] >= goldNeeded

EXAMPLE:
- Card cost: {{ "Blue": 5 }}, player bonuses: {{ "Blue": 0 }}, player gems: {{ "Blue": 1, "Gold": 1 }}
- effectiveCost = 5, shortage = 5 - 1 = 4, goldNeeded = 4
- player.gems[Gold] = 1 < 4 → CANNOT AFFORD → do NOT choose PURCHASE_CARD

NEVER suggest purchasing a card without completing all 4 steps above.

=== RULES ===
Each turn, choose EXACTLY ONE action:

1. TAKE_GEMS
   - Take 3 different colors (1 each), OR
   - Take 2 of the same color (bank must have >= 4 of that color)
   - Cannot take Gold directly
   - Total gems in hand must not exceed 10

2. PURCHASE_CARD
   - Buy a card from visibleCards (level1/2/3) or your own reservedCards
   - Must pass the AFFORD CHECK above before choosing this action

3. RESERVE_CARD
   - Reserve a visible card (max 3 reserved total)
   - Receive 1 Gold if bank has any
   - Use to block opponent or save for later

4. DISCARD_GEMS
   - Only when total gems would exceed 10 after TAKE_GEMS
   - Discard gems so total becomes exactly 10

5. SELECT_NOBLE
   - Only when you meet requirements of multiple nobles
   - Choose the noble that hurts opponent most

=== WINNING STRATEGY ===
- First to 15 points wins — always prioritize point efficiency
- Check opponent points: if they are at 12+ points → urgently buy high-point cards or block
- Nobles give 3 points each → track requirements and farm those colors
- level2/3 cards give more points than level1 — prioritize them
- Use RESERVE only to block opponent's winning card or secure a card you need
- Build bonuses (purchased cards) to reduce future costs — this is key to winning fast
- Never waste turns: every turn should move you closer to 15 points

=== OUTPUT ===
Return ONLY valid JSON, no other text:
{{
  "action": "TAKE_GEMS" | "PURCHASE_CARD" | "RESERVE_CARD" | "DISCARD_GEMS" | "SELECT_NOBLE",
  "payload": {{
    // TAKE_GEMS:     "gems": {{ "White": 1, "Blue": 1, "Green": 1 }}
    // PURCHASE_CARD: "cardId": "<cardId>"
    // RESERVE_CARD:  "cardId": "<cardId>"
    // DISCARD_GEMS:  "gems": {{ "White": 1, "Blue": 1 }}
    // SELECT_NOBLE:  "nobleId": "<nobleId>"
  }},
  "reasoning": "<short explanation including afford check result>"
}}`,
  ],
  [
    "human",
    `Current game state (it is {botId}'s turn):\n\`\`\`json\n{gameState}\n\`\`\`\nChoose the best action.`,
  ],
]);

const chain = prompt.pipe(llm).pipe(new StringOutputParser());

// ─── Schema ───────────────────────────────────────────────────
const ActionSchema = z.object({
  action: z.enum([
    "TAKE_GEMS",
    "PURCHASE_CARD",
    "RESERVE_CARD",
    "DISCARD_GEMS",
    "SELECT_NOBLE",
  ]),
  payload: z.record(z.string(), z.unknown()),
  reasoning: z.string(),
});

// ─── POST /decide ─────────────────────────────────────────────
app.post("/decide", async (req: Request, res: Response) => {
  try {
    const { gameState } = req.body;
    console.log(`gamestate`, gameState);
    if (!gameState) {
      return res.status(400).json({ error: "gameState is required" });
    }

    const botId: string = gameState?.turn?.currentPlayer ?? "BOT";
    const botState = gameState?.players?.[botId];
    const totalGems = botState?.gems
      ? Object.values(botState.gems as Record<string, number>).reduce(
          (a: number, b: number) => a + b,
          0,
        )
      : 0;

    console.log(
      `\n[Agent] Turn #${gameState?.turn?.turnNumber} | Bot: ${botId}`,
    );
    if (botState) {
      console.log(
        `        gems(${totalGems}): ${JSON.stringify(botState.gems)}`,
      );
      console.log(`        bonuses: ${JSON.stringify(botState.bonuses)}`);
      console.log(`        points: ${botState.points}`);
    }

    const raw = await chain.invoke({
      botId,
      gameState: JSON.stringify(gameState, null, 2),
    });

    const cleaned = raw.replace(/```json\n?|```\n?/g, "").trim();
    const result = ActionSchema.parse(JSON.parse(cleaned));

    console.log(`[Agent] → ${result.action}: ${result.reasoning}`);
    return res.json(result);
  } catch (err) {
    console.error("[Agent] Error:", err);
    return res.status(500).json({
      action: "TAKE_GEMS",
      payload: { gems: { White: 1, Blue: 1, Green: 1 } },
      reasoning: "fallback due to agent error",
    });
  }
});

// ─── Health check ─────────────────────────────────────────────
app.get("/health", (_: Request, res: Response) => res.json({ status: "ok" }));

// ─── Start ────────────────────────────────────────────────────
const PORT = process.env.PORT || 4000;
app.listen(Number(PORT), "0.0.0.0", () => {
  console.log(`🤖 Splendor LangChain Agent → http://localhost:${PORT}`);
  console.log(`   POST /decide  — receive gameState, return action`);
});
