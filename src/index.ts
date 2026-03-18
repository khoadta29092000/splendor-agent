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

// ─── Types ────────────────────────────────────────────────────
interface Gems {
  White?: number;
  Blue?: number;
  Green?: number;
  Red?: number;
  Black?: number;
  Gold?: number;
  [key: string]: number | undefined;
}
interface Card {
  cardId: string;
  level: number;
  points: number;
  bonusColor: string;
  cost: Gems;
}
interface Noble {
  nobleId: string;
  points: number;
  requirements: Gems;
}
interface Player {
  playerId: string;
  gems: Gems;
  bonuses: Gems;
  reservedCards: string[];
  purchasedCards: string[];
  points: number;
}
interface GameState {
  players: Record<string, Player>;
  board: {
    gemBank: Gems;
    visibleCards: { level1: Card[]; level2: Card[]; level3: Card[] };
    nobles: Noble[];
  };
  turn: { currentPlayer: string; turnNumber: number };
}
interface SuggestedAction {
  action: string;
  payload: Record<string, unknown>;
  score: number;
  reason: string;
}

// ─── Heuristic Helpers ────────────────────────────────────────
const COLORS = ["White", "Blue", "Green", "Red", "Black"];
const getGem = (gems: Gems, color: string): number => gems[color] ?? 0;
const totalGems = (gems: Gems): number =>
  Object.values(gems).reduce((a, b) => (a ?? 0) + (b ?? 0), 0) as number;

function affordCheck(card: Card, player: Player) {
  const shortage: Gems = {};
  let goldNeeded = 0;
  for (const color of COLORS) {
    const effective = Math.max(
      0,
      getGem(card.cost, color) - getGem(player.bonuses, color),
    );
    const short = Math.max(0, effective - getGem(player.gems, color));
    if (short > 0) {
      shortage[color] = short;
      goldNeeded += short;
    }
  }
  return {
    canAfford: getGem(player.gems, "Gold") >= goldNeeded,
    goldNeeded,
    shortage,
  };
}

function totalShortage(card: Card, player: Player): number {
  return Object.values(affordCheck(card, player).shortage).reduce(
    (a, b) => (a ?? 0) + (b ?? 0),
    0,
  ) as number;
}

function nearestNoble(player: Player, nobles: Noble[]) {
  if (!nobles?.length) return null;
  return nobles.reduce(
    (best, noble) => {
      const missing = COLORS.reduce(
        (sum, c) =>
          sum +
          Math.max(
            0,
            getGem(noble.requirements, c) - getGem(player.bonuses, c),
          ),
        0,
      );
      return !best || missing < best.missing ? { noble, missing } : best;
    },
    null as { noble: Noble; missing: number } | null,
  );
}

function colorsNeededForNoble(player: Player, nobles: Noble[]): string[] {
  const nearest = nearestNoble(player, nobles);
  if (!nearest) return [];
  return COLORS.filter(
    (c) => getGem(nearest.noble.requirements, c) > getGem(player.bonuses, c),
  );
}

function scoreCard(
  card: Card,
  player: Player,
  nobles: Noble[],
  opponentPoints: number,
): number {
  const { canAfford } = affordCheck(card, player);
  const shortage = totalShortage(card, player);
  const neededColors = colorsNeededForNoble(player, nobles);
  let score = card.points * 100 + card.level * 10 - shortage * 20;
  if (canAfford) score += 200;
  if (neededColors.includes(card.bonusColor)) score += 80;
  if (opponentPoints >= 12) score += card.points * 50;
  return score;
}

function getAllVisibleCards(state: GameState): Card[] {
  return [
    ...state.board.visibleCards.level1,
    ...state.board.visibleCards.level2,
    ...state.board.visibleCards.level3,
  ];
}

// ─── Suggestion Generators ────────────────────────────────────

function suggestPurchase(
  player: Player,
  state: GameState,
  opponentPoints: number,
): SuggestedAction[] {
  return getAllVisibleCards(state)
    .filter((c) => affordCheck(c, player).canAfford)
    .map((card) => ({
      action: "PURCHASE_CARD",
      payload: { cardId: card.cardId },
      score: scoreCard(card, player, state.board.nobles, opponentPoints) + 300,
      reason: `Buy lv${card.level} +${card.points}pts bonus:${card.bonusColor}`,
    }));
}

function suggestTakeGems(player: Player, state: GameState): SuggestedAction[] {
  const bank = state.board.gemBank;
  const current = totalGems(player.gems);
  const neededColors = colorsNeededForNoble(player, state.board.nobles);
  const suggestions: SuggestedAction[] = [];

  // Option A: 2 same color (bank >= 4)
  for (const color of COLORS) {
    if (getGem(bank, color) >= 4 && current + 2 <= 10) {
      suggestions.push({
        action: "TAKE_GEMS",
        payload: { gems: { [color]: 2 } },
        score: neededColors.includes(color) ? 150 : 80,
        reason: `Take 2 ${color} (bank=${getGem(bank, color)})`,
      });
    }
  }

  // Option B: 3 different colors
  const available = COLORS.filter((c) => getGem(bank, c) > 0);
  if (available.length >= 3 && current + 3 <= 10) {
    const prioritized = [
      ...neededColors.filter((c) => available.includes(c)),
      ...available.filter((c) => !neededColors.includes(c)),
    ].slice(0, 3);
    if (prioritized.length === 3) {
      const gems: Gems = {};
      prioritized.forEach((c) => (gems[c] = 1));
      suggestions.push({
        action: "TAKE_GEMS",
        payload: { gems },
        score: neededColors.some((c) => prioritized.includes(c)) ? 120 : 70,
        reason: `Take 3 different: ${prioritized.join(", ")}`,
      });
    }
  }

  // Option C: fallback khi bank gần cạn (< 3 màu available)
  if (
    available.length > 0 &&
    available.length < 3 &&
    current + available.length <= 10
  ) {
    const gems: Gems = {};
    available.forEach((c) => (gems[c] = 1));
    suggestions.push({
      action: "TAKE_GEMS",
      payload: { gems },
      score: 50,
      reason: `Take ${available.length} available: ${available.join(", ")}`,
    });
  }

  return suggestions;
}

function suggestReserve(
  player: Player,
  state: GameState,
  opponentPoints: number,
): SuggestedAction[] {
  if (player.reservedCards.length >= 3) return [];
  return getAllVisibleCards(state)
    .filter((c) => c.points >= 3)
    .sort((a, b) => b.points - a.points)
    .slice(0, 2)
    .map((card) => ({
      action: "RESERVE_CARD",
      payload: { cardId: card.cardId },
      score: opponentPoints >= 10 ? 100 : 30,
      reason: `Reserve lv${card.level} +${card.points}pts`,
    }));
}

function generateSuggestions(
  state: GameState,
  botId: string,
): SuggestedAction[] {
  const player = state.players[botId];
  if (!player) return [];
  const opponentId = Object.keys(state.players).find((id) => id !== botId);
  const opponentPoints = opponentId
    ? (state.players[opponentId]?.points ?? 0)
    : 0;
  return [
    ...suggestPurchase(player, state, opponentPoints),
    ...suggestTakeGems(player, state),
    ...suggestReserve(player, state, opponentPoints),
  ]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

// ─── Payload Validator ────────────────────────────────────────
// Validate và fix payload LLM trả về trước khi gửi tới BE

function validateAndFixPayload(
  result: {
    action: string;
    payload: Record<string, unknown>;
    reasoning: string;
  },
  suggestions: SuggestedAction[],
): { action: string; payload: Record<string, unknown>; reasoning: string } {
  const action = result.action;
  const payload = result.payload;

  switch (action) {
    case "TAKE_GEMS": {
      // Case 1: payload đúng format { gems: {...} }
      if (payload.gems && typeof payload.gems === "object") return result;

      // Case 2: LLM trả { White: 1, Blue: 1 } không có wrapper "gems"
      const colorKeys = COLORS.filter((c) => typeof payload[c] === "number");
      if (colorKeys.length > 0) {
        console.warn("[Validator] TAKE_GEMS missing 'gems' wrapper, fixing...");
        const gems: Record<string, number> = {};
        colorKeys.forEach((c) => (gems[c] = payload[c] as number));
        return { ...result, payload: { gems } };
      }

      // Case 3: payload hoàn toàn sai → dùng suggestion tốt nhất
      const bestTake = suggestions.find((s) => s.action === "TAKE_GEMS");
      if (bestTake) {
        console.warn(
          "[Validator] TAKE_GEMS invalid payload, using heuristic suggestion",
        );
        return {
          action: bestTake.action,
          payload: bestTake.payload,
          reasoning: bestTake.reason,
        };
      }
      break;
    }

    case "PURCHASE_CARD": {
      // Phải có cardId dạng string
      if (typeof payload.cardId === "string" && payload.cardId.length > 0)
        return result;

      // Fallback heuristic
      const bestPurchase = suggestions.find(
        (s) => s.action === "PURCHASE_CARD",
      );
      if (bestPurchase) {
        console.warn(
          "[Validator] PURCHASE_CARD invalid cardId, using heuristic suggestion",
        );
        return {
          action: bestPurchase.action,
          payload: bestPurchase.payload,
          reasoning: bestPurchase.reason,
        };
      }
      break;
    }

    case "RESERVE_CARD": {
      if (typeof payload.cardId === "string" && payload.cardId.length > 0)
        return result;

      const bestReserve = suggestions.find((s) => s.action === "RESERVE_CARD");
      if (bestReserve) {
        console.warn(
          "[Validator] RESERVE_CARD invalid cardId, using heuristic suggestion",
        );
        return {
          action: bestReserve.action,
          payload: bestReserve.payload,
          reasoning: bestReserve.reason,
        };
      }
      break;
    }

    case "DISCARD_GEMS": {
      if (payload.gems && typeof payload.gems === "object") return result;

      // Fix wrapper nếu thiếu
      const colorKeys = COLORS.filter((c) => typeof payload[c] === "number");
      if (colorKeys.length > 0) {
        const gems: Record<string, number> = {};
        colorKeys.forEach((c) => (gems[c] = payload[c] as number));
        return { ...result, payload: { gems } };
      }
      break;
    }

    case "SELECT_NOBLE": {
      if (typeof payload.nobleId === "string" && payload.nobleId.length > 0)
        return result;
      break;
    }
  }

  // Nếu không fix được → dùng top suggestion từ heuristic
  console.warn(
    `[Validator] Cannot fix ${action} payload, falling back to top suggestion`,
  );
  if (suggestions.length > 0) {
    return {
      action: suggestions[0].action,
      payload: suggestions[0].payload,
      reasoning: suggestions[0].reason,
    };
  }

  // Hard fallback
  return {
    action: "TAKE_GEMS",
    payload: { gems: { White: 1, Blue: 1, Green: 1 } },
    reasoning: "hard fallback",
  };
}

// ─── Prompt ───────────────────────────────────────────────────
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are playing Splendor. Goal: reach 15 prestige points first.
The Heuristic Engine has pre-calculated valid actions. Pick the BEST one.

Priority: 1) Buy card 2) Take gems for noble 3) Reserve to block

STRICT PAYLOAD RULE — copy EXACTLY from suggestions, do NOT restructure:
- TAKE_GEMS:    {{ "gems": {{ "White": 1, "Blue": 1, "Green": 1 }} }}  (3 diff colors)
               OR {{ "gems": {{ "Red": 2 }} }}  (2 same, only if bank >= 4)
- PURCHASE_CARD: {{ "cardId": "<id>" }}
- RESERVE_CARD:  {{ "cardId": "<id>" }}
- DISCARD_GEMS:  {{ "gems": {{ "White": 1 }} }}
- SELECT_NOBLE:  {{ "nobleId": "<id>" }}

Return ONLY valid JSON:
{{
  "action": "TAKE_GEMS"|"PURCHASE_CARD"|"RESERVE_CARD"|"DISCARD_GEMS"|"SELECT_NOBLE",
  "payload": {{ ... }},
  "reasoning": "<max 15 words>"
}}`,
  ],
  [
    "human",
    `Bot: {botId} | Turn: {turnNumber} | Points: {botPoints} | Gems({totalGems}): {botGems}
Opponent points: {opponentPoints} | {nobleProgress}

SUGGESTED ACTIONS (pre-validated, sorted by score):
{suggestions}

IMPORTANT: Copy payload EXACTLY as shown. Do NOT modify structure.
Choose the best action.`,
  ],
]);

const chain = prompt.pipe(llm).pipe(new StringOutputParser());
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
    const { gameState } = req.body as { gameState: GameState };
    if (!gameState)
      return res.status(400).json({ error: "gameState is required" });

    const botId = gameState?.turn?.currentPlayer ?? "BOT";
    const player = gameState?.players?.[botId];
    const opponentId = Object.keys(gameState.players).find(
      (id) => id !== botId,
    );
    const opponentPoints = opponentId
      ? (gameState.players[opponentId]?.points ?? 0)
      : 0;
    const totalGemsCount = player ? totalGems(player.gems) : 0;

    console.log(
      `\n[Agent] Turn #${gameState.turn?.turnNumber} | Bot: ${botId} | Points: ${player?.points} | Gems: ${totalGemsCount}`,
    );

    // Heuristic Engine
    const suggestions = generateSuggestions(gameState, botId);
    console.log(
      `[Heuristic] Suggestions:`,
      suggestions.map((s) => `${s.action}(${s.score})`).join(", "),
    );

    // Fast path: PURCHASE score cao → skip LLM
    if (
      suggestions[0]?.score >= 500 &&
      suggestions[0]?.action === "PURCHASE_CARD"
    ) {
      console.log(`[Heuristic] Fast purchase, skipping LLM`);
      return res.json({
        action: suggestions[0].action,
        payload: suggestions[0].payload,
        reasoning: suggestions[0].reason,
      });
    }

    // Fallback nếu không có suggestions
    if (!suggestions.length) {
      return res.json({
        action: "TAKE_GEMS",
        payload: { gems: { White: 1, Blue: 1, Green: 1 } },
        reasoning: "no suggestions",
      });
    }

    // Noble progress
    const nearest = player
      ? nearestNoble(player, gameState.board?.nobles ?? [])
      : null;
    const nobleProgress = nearest
      ? `Nearest noble: ${nearest.missing} gems away`
      : "No nobles";

    // LLM chọn từ suggestions
    const raw = await chain.invoke({
      botId,
      turnNumber: gameState.turn?.turnNumber,
      botPoints: player?.points ?? 0,
      totalGems: totalGemsCount,
      botGems: JSON.stringify(player?.gems ?? {}),
      opponentPoints,
      nobleProgress,
      suggestions: suggestions
        .map(
          (s, i) =>
            `${i + 1}. ${s.action} | ${JSON.stringify(s.payload)} | score:${s.score} | ${s.reason}`,
        )
        .join("\n"),
    });

    const cleaned = raw.replace(/```json\n?|```\n?/g, "").trim();
    const parsed = ActionSchema.parse(JSON.parse(cleaned));

    // ← Validate và fix payload trước khi gửi về BE
    const validated = validateAndFixPayload(parsed, suggestions);

    console.log(`[Agent] → ${validated.action}: ${validated.reasoning}`);
    return res.json(validated);
  } catch (err) {
    console.error("[Agent] Error:", err);
    return res.status(500).json({
      action: "TAKE_GEMS",
      payload: { gems: { White: 1, Blue: 1, Green: 1 } },
      reasoning: "fallback due to agent error",
    });
  }
});

app.get("/health", (_: Request, res: Response) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 4000;
console.log(`[Startup] PORT env = ${process.env.PORT}`);
console.log(`[Startup] Using PORT = ${PORT}`);

app.listen(Number(PORT), "0.0.0.0", () => {
  console.log(`🤖 Splendor LangChain Agent → http://localhost:${PORT}`);
  console.log(`   POST /decide  — receive gameState, return action`);
});
