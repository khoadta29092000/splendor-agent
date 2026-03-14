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
  modelName: "gpt-4o",
  temperature: 0.1,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// ─── Prompt ──────────────────────────────────────────────────
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an expert AI playing the board game Splendor.

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

=== RULES ===
Each turn, choose EXACTLY ONE action:

1. TAKE_GEMS
   - Take 3 different colors (1 each), OR
   - Take 2 of the same color (bank must have >= 4 of that color)
   - Cannot take Gold directly
   - Total gems in hand must not exceed 10
   - If taking gems would exceed 10, you MUST choose DISCARD_GEMS instead (see below)

2. PURCHASE_CARD
   - Buy a card from visibleCards (level1/2/3) or your own reservedCards
   - Effective cost: max(0, card.cost[color] - player.bonuses[color]) for each color
   - Can use Gold gems to cover remaining shortfall (1 Gold = 1 any color)

3. RESERVE_CARD
   - Reserve a visible card (max 3 reserved total)
   - Receive 1 Gold if bank has any
   - Use to block opponent or save for later purchase

4. DISCARD_GEMS
   - Required when total gems in hand would exceed 10 after TAKE_GEMS
   - Specify which gems to discard so total becomes exactly 10
   - payload: {{ "gems": {{ "White": 1, ... }} }} — gems to DISCARD

5. SELECT_NOBLE
   - Required when you meet requirements of multiple nobles after purchasing a card
   - Choose the noble that hurts the opponent most (they are closest to meeting)
   - payload: {{ "nobleId": "<nobleId>" }}

=== STRATEGY ===
- Always calculate effectiveCost correctly: max(0, cost[color] - bonuses[color])
- If you can afford a high-point card → buy it immediately
- Track nobles: if close to requirements → focus gems on that color
- Block opponent's high-point cards with RESERVE if they are 1-2 gems away
- Never pick gems if it would waste them (already have enough for target card)
- Prefer level2/3 cards with points over level1 cards

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
  "reasoning": "<short explanation>"
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
app.listen(PORT, () => {
  console.log(`🤖 Splendor LangChain Agent → http://localhost:${PORT}`);
  console.log(`   POST /decide  — receive gameState, return action`);
});
