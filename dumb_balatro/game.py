from random import Random
from enum import StrEnum, auto
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import NamedTuple

HAND_SIZE = 8
MAX_CARDS = 5
START_HANDS = 4
START_DISCARDS = 4


class Suit(StrEnum):
    SPADES = "Spades"
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"


class Rank(StrEnum):
    ACE = "Ace"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "Jack"
    QUEEN = "Queen"
    KING = "King"

    def number(self) -> int:
        match self:
            case Rank.TWO:
                return 2
            case Rank.THREE:
                return 3
            case Rank.FOUR:
                return 4
            case Rank.FIVE:
                return 5
            case Rank.SIX:
                return 6
            case Rank.SEVEN:
                return 7
            case Rank.EIGHT:
                return 8
            case Rank.NINE:
                return 9
            case Rank.TEN:
                return 10
            case Rank.JACK:
                return 11
            case Rank.QUEEN:
                return 12
            case Rank.KING:
                return 13
            case Rank.ACE:
                return 14

    def chips(self) -> int:
        match self:
            case Rank.TWO:
                return 2
            case Rank.THREE:
                return 3
            case Rank.FOUR:
                return 4
            case Rank.FIVE:
                return 5
            case Rank.SIX:
                return 6
            case Rank.SEVEN:
                return 7
            case Rank.EIGHT:
                return 8
            case Rank.NINE:
                return 9
            case Rank.TEN | Rank.JACK | Rank.QUEEN | Rank.KING:
                return 10
            case Rank.ACE:
                return 11


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def chips(self) -> int:
        return self.rank.chips()

    def __repr__(self) -> str:
        return f"({self.suit}, {self.rank})"

    def __str__(self) -> str:
        suit_idx = list(Suit).index(self.suit)
        rank_idx = list(Rank).index(self.rank)
        if rank_idx >= 0xB:
            rank_idx += 1
        card_start = 0x1F0A1
        return chr(card_start + 0x10 * suit_idx + rank_idx)


class Scoring(NamedTuple):
    chips: int
    mult: int


class HandType(StrEnum):
    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()

    def base_scoring(self) -> Scoring:
        match self:
            case HandType.HIGH_CARD:
                return Scoring(chips=5, mult=1)
            case HandType.PAIR:
                return Scoring(chips=10, mult=2)
            case HandType.TWO_PAIR:
                return Scoring(chips=20, mult=2)
            case HandType.THREE_OF_A_KIND:
                return Scoring(chips=30, mult=3)
            case HandType.STRAIGHT:
                return Scoring(chips=30, mult=4)
            case HandType.FLUSH:
                return Scoring(chips=35, mult=4)
            case HandType.FULL_HOUSE:
                return Scoring(chips=40, mult=4)
            case HandType.FOUR_OF_A_KIND:
                return Scoring(chips=60, mult=7)
            case HandType.STRAIGHT_FLUSH:
                return Scoring(chips=100, mult=8)


def count_aggregator[T](arr: list[T]) -> dict[T, list[int]]:
    counts = defaultdict(list)
    for i, elem in enumerate(arr):
        counts[elem].append(i)
    return counts


class Hand:
    hand_type: HandType
    indices: list[int]

    def __init__(self, hand_type, indices) -> None:
        self.hand_type = hand_type
        self.indices = indices

    @classmethod
    def __check_straights(cls, ranks, rank_order) -> list[int] | None:
        sorted_ranks = sorted(rank_order[rank] for rank in set(ranks))
        # Look for consecutive sequences
        indices = []
        for i in range(0, len(sorted_ranks) - 1):
            if sorted_ranks[i] + 1 == sorted_ranks[i + 1]:
                indices.append(i)
                if len(indices) >= 4:
                    indices.append(i + 1)
                    return indices
            else:
                indices = []
        return None

    @classmethod
    def classify_hand(cls, hand: list[Card]) -> "Hand":
        ranks = [card.rank for card in hand]
        suits = [card.suit for card in hand]
        rank_counts = count_aggregator(ranks)
        suit_counts = count_aggregator(suits)

        # Determine hand type based on a hierarchy:
        # Full house, flush, straight, three-of-a-kind, two pair, pair, high card

        # straights
        # ace-high
        rank_order = {rank: rank.number() for rank in Rank}
        straight = cls.__check_straights(ranks, rank_order)
        # ace-low
        rank_order[Rank.ACE] = 1
        if not straight:
            straight = cls.__check_straights(ranks, rank_order)

        # flush
        flush = next(
            (indices for indices in suit_counts.values() if len(indices) >= 5), None
        )

        if straight and all(suits[straight[0]] == suits[i] for i in straight):
            return cls(HandType.STRAIGHT_FLUSH, straight)
        elif straight:
            return cls(HandType.STRAIGHT, straight)
        elif flush:
            return cls(HandType.FLUSH, flush)

        # full house: one three-of-a-kind and one pair
        pairs = [indices for indices in rank_counts.values() if len(indices) == 2]
        threes = next(
            (indices for indices in rank_counts.values() if len(indices) == 3), None
        )
        if len(pairs) > 0 and threes:
            return cls(HandType.FULL_HOUSE, pairs[0] + threes)
        # three-of-a-kind
        elif threes:
            return cls(HandType.THREE_OF_A_KIND, threes)
        # Check for two pair (only if not full house)
        elif len(pairs) >= 2:
            return cls(HandType.TWO_PAIR, [i for pair in pairs for i in pair])
        elif len(pairs) >= 1:
            return cls(HandType.PAIR, [i for pair in pairs for i in pair])
        else:
            return cls(
                HandType.HIGH_CARD,
                [max(range(len(hand)), key=lambda i: ranks[i].number())],
            )


class InvalidPlayType(StrEnum):
    TOO_MANY_CARDS = auto()
    DUPLICATE_CARDS = auto()
    INVALID_INDICES = auto()
    NOT_ENOUGH_DISCARDS = auto()
    NOT_ENOUGH_HANDS = auto()


class InvalidPlay(Exception):
    def __init__(self, play_type: InvalidPlayType) -> None:
        super().__init__(str(play_type))


class DumbBalatro:
    random: Random
    deck: list[Card]
    current_deck: list[Card]
    hand: list[Card]
    hands: int
    discards: int

    def __init__(self, seed: int | None = None) -> None:
        self.deck = [Card(suit, rank) for suit in Suit for rank in Rank]
        self.hand = []
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.random = Random(seed)
        self.hands = START_HANDS
        self.discards = START_DISCARDS
        self.shuffle_deck()
        self.draw()

    def shuffle_deck(self) -> None:
        self.current_deck = self.deck.copy()
        self.random.shuffle(self.current_deck)

    def draw(self) -> list[Card]:
        for _ in range(min(HAND_SIZE - len(self.hand), len(self.deck))):
            self.hand.append(self.current_deck.pop())
        return self.hand

    def is_ended(self) -> bool:
        return self.hands <= 0

    def validate_play(self, indices: list[int], discard: bool) -> None:
        if discard and self.discards <= 0:
            raise InvalidPlay(InvalidPlayType.NOT_ENOUGH_DISCARDS)
        elif self.hands <= 0:
            raise InvalidPlay(InvalidPlayType.NOT_ENOUGH_HANDS)
        if len(indices) > 5:
            raise InvalidPlay(InvalidPlayType.TOO_MANY_CARDS)
        if any(i >= len(self.hand) for i in indices):
            raise InvalidPlay(InvalidPlayType.INVALID_INDICES)
        cards = [self.hand[i] for i in indices]
        counts = Counter(cards)
        if any(count > 1 for count in counts.values()):
            raise InvalidPlay(InvalidPlayType.DUPLICATE_CARDS)

    def score(self, cards: list[Card]) -> int:
        hand = Hand.classify_hand(cards)
        chips, mult = hand.hand_type.base_scoring()
        chips += sum(cards[i].chips() for i in hand.indices)
        return chips * mult

    def play(self, indices: list[int], discard=False) -> int | None:
        self.validate_play(indices, discard)
        cards = [self.hand[i] for i in indices]
        self.hand = [c for i, c in enumerate(self.hand) if i not in indices]
        self.draw()
        if discard:
            self.discards -= 1
            return None
        else:
            self.hands -= 1
            return self.score(cards)


if __name__ == "__main__":
    balatro = DumbBalatro(seed=1234)
    while not balatro.is_ended():
        print(balatro.hand)
        print(f"indx: {' '.join(str(x) for x in range(len(balatro.hand)))}")
        print(f"hand: {' '.join(str(card) for card in balatro.hand)}")
        play = input("play: ").split()
        discard = False
        if play[-1] == "d":
            discard = True
        print(balatro.play([int(idx) for idx in play], discard))
