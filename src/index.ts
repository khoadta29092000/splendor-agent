import "dotenv/config";
import express, { Request, Response } from "express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

const app = express();
app.use(express.json({ limit: "1mb" }));

const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0.1,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

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

const COLORS = ["White", "Blue", "Green", "Red", "Black"];
const getGem = (gems: Gems, color: string): number => gems[color] ?? 0;
const totalGems = (gems: Gems): number =>
  Object.values(gems).reduce((a, b) => (a ?? 0) + (b ?? 0), 0) as number;

// Kiểm tra player có đủ gem/bonus để mua card không, trả về lượng gold cần bù
function affordCheck(card: Card, player: Player) {
  const shortage: Gems = {};
  let goldNeeded = 0;
  for (const color of COLORS) {
    const effective = Math.max(0, getGem(card.cost, color) - getGem(player.bonuses, color));
    const short = Math.max(0, effective - getGem(player.gems, color));
    if (short > 0) {
      shortage[color] = short;
      goldNeeded += short;
    }
  }
  return { canAfford: getGem(player.gems, "Gold") >= goldNeeded, goldNeeded, shortage };
}

// Tổng số gem còn thiếu để mua card
function totalShortage(card: Card, player: Player): number {
  return Object.values(affordCheck(card, player).shortage).reduce(
    (a, b) => (a ?? 0) + (b ?? 0), 0,
  ) as number;
}

// Tìm noble gần đạt nhất dựa trên bonus hiện tại của player
function nearestNoble(player: Player, nobles: Noble[]) {
  if (!nobles?.length) return null;
  return nobles.reduce(
    (best, noble) => {
      const missing = COLORS.reduce(
        (sum, c) => sum + Math.max(0, getGem(noble.requirements, c) - getGem(player.bonuses, c)), 0,
      );
      return !best || missing < best.missing ? { noble, missing } : best;
    },
    null as { noble: Noble; missing: number } | null,
  );
}

// Trả về danh sách màu bonus còn thiếu để đạt noble gần nhất
function colorsNeededForNoble(player: Player, nobles: Noble[]): string[] {
  const nearest = nearestNoble(player, nobles);
  if (!nearest) return [];
  return COLORS.filter((c) => getGem(nearest.noble.requirements, c) > getGem(player.bonuses, c));
}

// Tính score ưu tiên cho card: điểm cao + mua được + hướng noble + chặn đối thủ
function scoreCard(card: Card, player: Player, nobles: Noble[], opponentPoints: number): number {
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

// Gợi ý mua card: chỉ card mua được, ưu tiên điểm cao
function suggestPurchase(player: Player, state: GameState, opponentPoints: number): SuggestedAction[] {
  return getAllVisibleCards(state)
    .filter((c) => affordCheck(c, player).canAfford)
    .map((card) => ({
      action: "PURCHASE_CARD",
      payload: { cardId: card.cardId },
      score: scoreCard(card, player, state.board.nobles, opponentPoints) + 300,
      reason: `Buy lv${card.level} +${card.points}pts bonus:${card.bonusColor}`,
    }));
}

// Gợi ý lấy gem theo rule Splendor:
// Option A: 2 cùng màu nếu bank >= 4
// Option B: 3 màu khác nhau nếu có >= 3 màu available
// Option C: 2 màu khác nhau nếu chỉ còn 2 màu | 2 cùng màu nếu chỉ còn 1 màu và bank >= 4
function suggestTakeGems(player: Player, state: GameState): SuggestedAction[] {
  const bank = state.board.gemBank;
  const current = totalGems(player.gems);
  const neededColors = colorsNeededForNoble(player, state.board.nobles);
  const suggestions: SuggestedAction[] = [];

  // Option A
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

  const available = COLORS.filter((c) => getGem(bank, c) > 0);

  // Option B
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

  // Option C
  if (available.length === 2 && current + 2 <= 10) {
    const gems: Gems = {};
    available.forEach((c) => (gems[c] = 1));
    suggestions.push({
      action: "TAKE_GEMS",
      payload: { gems },
      score: 50,
      reason: `Take 2 available: ${available.join(", ")}`,
    });
  } else if (available.length === 1 && getGem(bank, available[0]) >= 4 && current + 2 <= 10) {
    suggestions.push({
      action: "TAKE_GEMS",
      payload: { gems: { [available[0]]: 2 } },
      score: 50,
      reason: `Take 2 ${available[0]} (only color left, bank=${getGem(bank, available[0])})`,
    });
  }

  return suggestions;
}

// Gợi ý reserve card: ưu tiên card điểm cao, bỏ qua nếu đã có 3 reserved
function suggestReserve(player: Player, state: GameState, opponentPoints: number): SuggestedAction[] {
  if (player.reservedCards.length >= 3) return [];
  const allCards = getAllVisibleCards(state);
  if (allCards.length === 0) return [];
  return allCards
    .sort((a, b) => b.points - a.points)
    .slice(0, 2)
    .map((card) => ({
      action: "RESERVE_CARD",
      payload: { cardId: card.cardId },
      score: opponentPoints >= 10 ? 100 : 40,
      reason: `Reserve lv${card.level} +${card.points}pts`,
    }));
}

// Tổng hợp tất cả suggestions, fallback PASS_TURN nếu không có action hợp lệ nào
function generateSuggestions(state: GameState, botId: string): SuggestedAction[] {
  const player = state.players[botId];
  if (!player) return [];
  const opponentId = Object.keys(state.players).find((id) => id !== botId);
  const opponentPoints = opponentId ? (state.players[opponentId]?.points ?? 0) : 0;

  const suggestions = [
    ...suggestPurchase(player, state, opponentPoints),
    ...suggestTakeGems(player, state),
    ...suggestReserve(player, state, opponentPoints),
  ]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  if (suggestions.length === 0) {
    suggestions.push({
      action: "PASS_TURN",
      payload: {},
      score: 0,
      reason: "No valid actions available",
    });
  }

  return suggestions;
}

// Validate và fix payload LLM trả về:
// - Fix format sai (thiếu wrapper gems, cardId rỗng...)
// - Validate gem có thực sự còn trên board không
// - Fallback về heuristic suggestion nếu không fix được
function validateAndFixPayload(
  result: { action: string; payload: Record<string, unknown>; reasoning: string },
  suggestions: SuggestedAction[],
  gameState: GameState,
): { action: string; payload: Record<string, unknown>; reasoning: string } {
  const action = result.action;
  const payload = result.payload;
  const bank = gameState.board.gemBank;

  switch (action) {
    case "TAKE_GEMS": {
      if (payload.gems && typeof payload.gems === "object") {
        const requestedGems = payload.gems as Record<string, number>;
        const isValid = Object.entries(requestedGems).every(
          ([color, amount]) => (bank[color] ?? 0) >= amount,
        );
        if (isValid) return result;
        console.warn("[Validator] TAKE_GEMS gems not available on board, using heuristic");
      } else {
        const colorKeys = COLORS.filter((c) => typeof payload[c] === "number");
        if (colorKeys.length > 0) {
          console.warn("[Validator] TAKE_GEMS missing 'gems' wrapper, fixing...");
          const gems: Record<string, number> = {};
          colorKeys.forEach((c) => (gems[c] = payload[c] as number));
          const isValid = Object.entries(gems).every(([color, amount]) => (bank[color] ?? 0) >= amount);
          if (isValid) return { ...result, payload: { gems } };
        }
      }

      const bestTake = suggestions.find((s) => s.action === "TAKE_GEMS");
      if (bestTake) return { action: bestTake.action, payload: bestTake.payload, reasoning: bestTake.reason };

      const bestReserve = suggestions.find((s) => s.action === "RESERVE_CARD");
      if (bestReserve) return { action: bestReserve.action, payload: bestReserve.payload, reasoning: bestReserve.reason };

      return { action: "PASS_TURN", payload: {}, reasoning: "no gems available" };
    }

    case "PURCHASE_CARD": {
      if (typeof payload.cardId === "string" && payload.cardId.length > 0) return result;
      const bestPurchase = suggestions.find((s) => s.action === "PURCHASE_CARD");
      if (bestPurchase) {
        console.warn("[Validator] PURCHASE_CARD invalid cardId, using heuristic suggestion");
        return { action: bestPurchase.action, payload: bestPurchase.payload, reasoning: bestPurchase.reason };
      }
      break;
    }

    case "RESERVE_CARD": {
      if (typeof payload.cardId === "string" && payload.cardId.length > 0) return result;
      const bestReserve = suggestions.find((s) => s.action === "RESERVE_CARD");
      if (bestReserve) {
        console.warn("[Validator] RESERVE_CARD invalid cardId, using heuristic suggestion");
        return { action: bestReserve.action, payload: bestReserve.payload, reasoning: bestReserve.reason };
      }
      break;
    }

    case "DISCARD_GEMS": {
      if (payload.gems && typeof payload.gems === "object") return result;
      const colorKeys = COLORS.filter((c) => typeof payload[c] === "number");
      if (colorKeys.length > 0) {
        const gems: Record<string, number> = {};
        colorKeys.forEach((c) => (gems[c] = payload[c] as number));
        return { ...result, payload: { gems } };
      }
      break;
    }

    case "SELECT_NOBLE": {
      if (typeof payload.nobleId === "string" && payload.nobleId.length > 0) return result;
      break;
    }

    case "PASS_TURN": {
      return { action: "PASS_TURN", payload: {}, reasoning: result.reasoning };
    }
  }

  console.warn(`[Validator] Cannot fix ${action} payload, falling back to top suggestion`);
  if (suggestions.length > 0) {
    return { action: suggestions[0].action, payload: suggestions[0].payload, reasoning: suggestions[0].reason };
  }

  return { action: "PASS_TURN", payload: {}, reasoning: "hard fallback" };
}

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are playing Splendor. Goal: reach 15 prestige points first.
The Heuristic Engine has pre-calculated valid actions. Pick the BEST one.

Priority: 1) Buy card 2) Take gems for noble 3) Reserve to block 4) Pass turn

STRICT PAYLOAD RULE — copy EXACTLY from suggestions, do NOT restructure:
- TAKE_GEMS:     {{ "gems": {{ "White": 1, "Blue": 1, "Green": 1 }} }}  (3 diff colors)
                OR {{ "gems": {{ "Red": 2 }} }}  (2 same, only if bank >= 4)
- PURCHASE_CARD: {{ "cardId": "<id>" }}
- RESERVE_CARD:  {{ "cardId": "<id>" }}
- DISCARD_GEMS:  {{ "gems": {{ "White": 1 }} }}
- SELECT_NOBLE:  {{ "nobleId": "<id>" }}
- PASS_TURN:     {{}}  (only when no valid action exists)

Return ONLY valid JSON:
{{
  "action": "TAKE_GEMS"|"PURCHASE_CARD"|"RESERVE_CARD"|"DISCARD_GEMS"|"SELECT_NOBLE"|"PASS_TURN",
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
    "PASS_TURN",
  ]),
  payload: z.record(z.string(), z.unknown()),
  reasoning: z.string(),
});

app.post("/decide", async (req: Request, res: Response) => {
  try {
    const { gameState } = req.body as { gameState: GameState };
    if (!gameState) return res.status(400).json({ error: "gameState is required" });

    const botId = gameState?.turn?.currentPlayer ?? "BOT";
    const player = gameState?.players?.[botId];
    const opponentId = Object.keys(gameState.players).find((id) => id !== botId);
    const opponentPoints = opponentId ? (gameState.players[opponentId]?.points ?? 0) : 0;
    const totalGemsCount = player ? totalGems(player.gems) : 0;

    console.log(`\n[Agent] Turn #${gameState.turn?.turnNumber} | Bot: ${botId} | Points: ${player?.points} | Gems: ${totalGemsCount}`);

    const suggestions = generateSuggestions(gameState, botId);
    console.log(`[Heuristic] Suggestions:`, suggestions.map((s) => `${s.action}(${s.score})`).join(", "));

    // Fast path: mua được card điểm cao → skip LLM
    if (suggestions[0]?.score >= 500 && suggestions[0]?.action === "PURCHASE_CARD") {
      console.log(`[Heuristic] Fast purchase, skipping LLM`);
      return res.json({ action: suggestions[0].action, payload: suggestions[0].payload, reasoning: suggestions[0].reason });
    }

    // Fast path: không có action nào hợp lệ
    if (suggestions[0]?.action === "PASS_TURN") {
      console.log(`[Heuristic] No valid actions, passing turn`);
      return res.json({ action: "PASS_TURN", payload: {}, reasoning: "no valid actions" });
    }

    const nearest = player ? nearestNoble(player, gameState.board?.nobles ?? []) : null;
    const nobleProgress = nearest ? `Nearest noble: ${nearest.missing} gems away` : "No nobles";

    const raw = await chain.invoke({
      botId,
      turnNumber: gameState.turn?.turnNumber,
      botPoints: player?.points ?? 0,
      totalGems: totalGemsCount,
      botGems: JSON.stringify(player?.gems ?? {}),
      opponentPoints,
      nobleProgress,
      suggestions: suggestions
        .map((s, i) => `${i + 1}. ${s.action} | ${JSON.stringify(s.payload)} | score:${s.score} | ${s.reason}`)
        .join("\n"),
    });

    const cleaned = raw.replace(/```json\n?|```\n?/g, "").trim();
    const parsed = ActionSchema.parse(JSON.parse(cleaned));
    const validated = validateAndFixPayload(parsed, suggestions, gameState);

    console.log(`[Agent] → ${validated.action}: ${validated.reasoning}`);
    return res.json(validated);
  } catch (err) {
    console.error("[Agent] Error:", err);
    return res.status(500).json({ action: "PASS_TURN", payload: {}, reasoning: "fallback due to agent error" });
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
