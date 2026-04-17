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
  temperature: 0, // FIX: 0 thay vì 0.1 — game logic cần deterministic
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// ============================================================================
// INTERFACES — DecisionContext là input mới từ C# (pre-computed)
// ============================================================================

interface Gems {
  White?: number;
  Blue?: number;
  Green?: number;
  Red?: number;
  Black?: number;
  Gold?: number;
  [key: string]: number | undefined;
}

// Card với affordability đã tính sẵn từ C#
interface CardWithAffordability {
  cardId: string;
  level: number;
  points: number;
  bonusColor: string;
  cost: Gems;
  shortfall: Gems; // C# pre-computed
  goldNeeded: number; // C# pre-computed
  canAfford: boolean; // C# pre-computed
  isReserved: boolean;
}

interface NobleWithProgress {
  nobleId: string;
  points: number;
  missingBonuses: Gems; // C# pre-computed
  totalMissing: number; // C# pre-computed
}

// Input mới từ C# BuildStrategyPayload
interface DecisionContext {
  turn: { number: number };
  bot: {
    points: number;
    gems: Gems;
    totalGems: number; // C# pre-computed
    bonuses: Gems;
    reservedCount: number;
  };
  opponent: {
    points: number;
    bonuses: Gems;
    reservedCount: number;
  };
  board: {
    gemBank: Gems;
    nobles: NobleWithProgress[];
  };
  cards: CardWithAffordability[]; // visible + reserved, pre-computed
  computed: {
    canBuyAny: boolean;
    nearestNoble: NobleWithProgress | null;
    isEndgame: boolean;
  };
}

// Input cũ /decide — vẫn giữ để backward compat nếu cần
interface GameState {
  players: Record<
    string,
    {
      playerId: string;
      gems: Gems;
      bonuses: Gems;
      reservedCards: string[];
      purchasedCards: string[];
      points: number;
    }
  >;
  board: {
    gemBank: Gems;
    visibleCards: { level1: any[]; level2: any[]; level3: any[] };
    nobles: { nobleId: string; points: number; requirements: Gems }[];
  };
  turn: { currentPlayer: string; turnNumber: number };
}

const COLORS = ["White", "Blue", "Green", "Red", "Black"];
const getGem = (gems: Gems, c: string): number => gems[c] ?? 0;

// ============================================================================
// /strategy ENDPOINT — nhận DecisionContext, trả targetCardId
// ============================================================================

const strategyPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a Splendor AI strategist. Goal: reach 15 prestige points first.
All affordability and shortfall data is pre-computed — DO NOT recalculate.

STRATEGY RULES (in order):
1. If canBuyAny=true → pick the highest-value affordable card as target
2. Prefer cards that align with nearest noble's missing bonuses
3. Prefer lower goldNeeded (closer to affordable) when points are equal
4. In endgame (isEndgame=true) → maximize points per turn, ignore noble

Return ONLY valid JSON — no markdown, no preamble:
{{
  "targetCardId": "<cardId from cards list, or null>",
  "targetNobleId": "<nobleId from nobles list, or null>",
  "reasoning": "<max 20 words>"
}}`,
  ],
  [
    "human",
    `Turn: {turnNumber} | My points: {myPoints} | Opponent: {opponentPoints}pts | Endgame: {isEndgame}
My gems (total={totalGems}): {myGems}
My bonuses: {myBonuses}
Nearest noble: {nearestNoble}

CARDS (pre-computed, visible + reserved):
{cards}

Pick the best target card to farm gems toward.`,
  ],
]);

const strategyChain = strategyPrompt.pipe(llm).pipe(new StringOutputParser());

const StrategySchema = z.object({
  targetCardId: z.string().nullable(),
  targetNobleId: z.string().nullable(),
  reasoning: z.string(),
});

app.post("/strategy", async (req: Request, res: Response) => {
  try {
    const ctx = req.body as DecisionContext;
    if (!ctx?.cards)
      return res.status(400).json({ error: "DecisionContext required" });

    const { bot, opponent, computed, board } = ctx;

    console.log(
      `\n[Strategy] Turn #${ctx.turn.number} | Bot ${bot.points}pts | Opp ${opponent.points}pts | Endgame=${computed.isEndgame}`,
    );

    // Fast path: có card mua được với điểm cao → target ngay
    const fastTarget = ctx.cards
      .filter((c) => c.canAfford && c.points >= 3)
      .sort(
        (a, b) =>
          b.points * 100 + b.level * 10 - (a.points * 100 + a.level * 10),
      )[0];

    if (fastTarget) {
      console.log(
        `[Strategy] FastTarget ${fastTarget.cardId} lv${fastTarget.level} +${fastTarget.points}pts`,
      );
      return res.json({
        targetCardId: fastTarget.cardId,
        targetNobleId: computed.nearestNoble?.nobleId ?? null,
        reasoning: `Fast target: lv${fastTarget.level} +${fastTarget.points}pts already affordable`,
      });
    }

    // Prompt cards — chỉ relevant info, không có imageUrl
    const cardsStr = ctx.cards
      .map((c) => {
        const shortfallStr = Object.entries(c.shortfall)
          .map(([k, v]) => `${k}:${v}`)
          .join(",");
        return (
          `[${c.cardId}] lv${c.level} +${c.points}pts bonus:${c.bonusColor}` +
          ` canAfford:${c.canAfford} goldNeeded:${c.goldNeeded}` +
          ` shortfall:{${shortfallStr}}` +
          (c.isReserved ? " [RESERVED]" : "")
        );
      })
      .join("\n");

    const nearestNobleStr = computed.nearestNoble
      ? `${computed.nearestNoble.nobleId} totalMissing:${computed.nearestNoble.totalMissing} missing:${JSON.stringify(computed.nearestNoble.missingBonuses)}`
      : "none";

    const raw = await strategyChain.invoke({
      turnNumber: ctx.turn.number,
      myPoints: bot.points,
      opponentPoints: opponent.points,
      isEndgame: computed.isEndgame,
      totalGems: bot.totalGems,
      myGems: JSON.stringify(bot.gems),
      myBonuses: JSON.stringify(bot.bonuses),
      nearestNoble: nearestNobleStr,
      cards: cardsStr,
    });

    const cleaned = raw.replace(/```json\n?|```\n?/g, "").trim();
    const parsed = StrategySchema.parse(JSON.parse(cleaned));

    // Validate targetCardId tồn tại trong card list
    if (parsed.targetCardId) {
      const exists = ctx.cards.some((c) => c.cardId === parsed.targetCardId);
      if (!exists) {
        console.warn(
          `[Strategy] LLM returned unknown cardId=${parsed.targetCardId}, using heuristic`,
        );
        // Fallback: card gần mua nhất + điểm cao
        const fallback = ctx.cards.sort((a, b) => {
          const scoreA = a.points * 100 - a.goldNeeded * 20;
          const scoreB = b.points * 100 - b.goldNeeded * 20;
          return scoreB - scoreA;
        })[0];
        parsed.targetCardId = fallback?.cardId ?? null;
      }
    }

    console.log(
      `[Strategy] → target=${parsed.targetCardId} noble=${parsed.targetNobleId} | ${parsed.reasoning}`,
    );
    return res.json(parsed);
  } catch (err) {
    console.error("[Strategy] Error:", err);
    // Fallback heuristic nếu LLM fail
    const ctx = req.body as DecisionContext;
    const fallback = ctx?.cards?.sort(
      (a, b) =>
        b.points * 100 -
        b.goldNeeded * 20 -
        (a.points * 100 - a.goldNeeded * 20),
    )[0];
    return res.json({
      targetCardId: fallback?.cardId ?? null,
      targetNobleId: null,
      reasoning: "fallback heuristic",
    });
  }
});

// ============================================================================
// /decide ENDPOINT — legacy, giữ để backward compat
// Nếu C# đã dùng LangChainBotService mới thì endpoint này không còn được gọi.
// ============================================================================

app.post("/decide", async (req: Request, res: Response) => {
  try {
    const { gameState } = req.body as { gameState: GameState };
    if (!gameState)
      return res.status(400).json({ error: "gameState is required" });

    const botId = gameState?.turn?.currentPlayer ?? "BOT";
    const player = gameState?.players?.[botId];
    if (!player) return res.status(400).json({ error: "bot player not found" });

    const opponentId = Object.keys(gameState.players).find(
      (id) => id !== botId,
    );
    const opponentPoints = opponentId
      ? (gameState.players[opponentId]?.points ?? 0)
      : 0;
    const currentGems = Object.values(player.gems).reduce(
      (a, b) => (a ?? 0) + (b ?? 0),
      0,
    ) as number;

    console.log(
      `\n[Decide/Legacy] Turn #${gameState.turn?.turnNumber} | Bot ${player.points}pts | Gems ${currentGems}`,
    );

    // FIX: Cho phép lấy gem kể cả khi currentGems >= 10 — BE sẽ trigger discard
    const allCards = [
      ...gameState.board.visibleCards.level1,
      ...gameState.board.visibleCards.level2,
      ...gameState.board.visibleCards.level3,
    ];
    const bank = gameState.board.gemBank;

    // Tìm card tốt nhất mua được
    const affordable = allCards
      .filter((c) => {
        let goldNeeded = 0;
        for (const color of COLORS) {
          const eff = Math.max(
            0,
            getGem(c.cost, color) - getGem(player.bonuses, color),
          );
          const diff = Math.max(0, eff - getGem(player.gems, color));
          goldNeeded += diff;
        }
        return getGem(player.gems, "Gold") >= goldNeeded;
      })
      .sort(
        (a, b) =>
          b.points * 100 + b.level * 10 - (a.points * 100 + a.level * 10),
      );

    if (affordable.length > 0) {
      return res.json({
        action: "PURCHASE_CARD",
        payload: { cardId: affordable[0].cardId },
        reasoning: `Buy lv${affordable[0].level} +${affordable[0].points}pts`,
      });
    }

    // FIX: Lấy gem — không block khi currentGems >= 10
    const available = COLORS.filter((c) => getGem(bank, c) > 0);
    if (available.length >= 3) {
      const gems: Gems = {};
      available.slice(0, 3).forEach((c) => (gems[c] = 1));
      return res.json({
        action: "TAKE_GEMS",
        payload: { gems },
        reasoning: "Take 3 different gems",
      });
    }
    if (available.length === 2) {
      const gems: Gems = {};
      available.forEach((c) => (gems[c] = 1));
      return res.json({
        action: "TAKE_GEMS",
        payload: { gems },
        reasoning: "Take 2 available gems",
      });
    }
    if (available.length === 1 && getGem(bank, available[0]) >= 4) {
      return res.json({
        action: "TAKE_GEMS",
        payload: { gems: { [available[0]]: 2 } },
        reasoning: `Take 2x ${available[0]}`,
      });
    }

    return res.json({
      action: "PASS_TURN",
      payload: {},
      reasoning: "no valid actions",
    });
  } catch (err) {
    console.error("[Decide/Legacy] Error:", err);
    return res
      .status(500)
      .json({ action: "PASS_TURN", payload: {}, reasoning: "error fallback" });
  }
});

// ============================================================================
// HEALTH
// ============================================================================

app.get("/health", (_: Request, res: Response) =>
  res.json({ status: "ok", endpoints: ["/strategy", "/decide", "/health"] }),
);

const PORT = process.env.PORT || 4000;
app.listen(Number(PORT), "0.0.0.0", () => {
  console.log(`Splendor Strategy Agent → http://localhost:${PORT}`);
  console.log(
    `  POST /strategy — nhận DecisionContext, trả targetCardId (LangChainBotService mới)`,
  );
  console.log(`  POST /decide   — legacy endpoint (LangChainBotService cũ)`);
});
