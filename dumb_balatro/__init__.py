from enum import StrEnum, auto
from dataclasses import dataclass
from itertools import product
from collections import Counter

HAND_SIZE = 8
MAX_CARDS = 5


class Suit(StrEnum):
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    HEARTS = "Hearts"
    SPADES = "Spades"


class Rank(StrEnum):
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
    ACE = "Ace"


@dataclass
class Card:
    suit: Suit
    rank: Rank


@dataclass
class Scoring:
    chips: int
    mult: int


class Hand(StrEnum):
    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()
    ROYAL_FLUSH = auto()

    @classmethod
    def __check_straights(cls, ranks, rank_order):
        sorted_ranks = sorted(rank_order[rank] for rank in set(ranks))
        # Look for consecutive sequences
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 5:
                    return True
            else:
                consecutive = 1
        return False

    @classmethod
    def classify_hand(cls, hand: list[Card]) -> "Hand":
        ranks = [card.rank for card in hand]
        suits = [card.suit for card in hand]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Determine hand type based on a hierarchy:
        # Full house, flush, straight, three-of-a-kind, two pair, pair, high card

        # straights
        straight = False
        # ace-high
        rank_order = {r: i for i, r in enumerate(Rank, start=2)}
        if cls.__check_straights(ranks, rank_order):
            straight = True
        # ace-low
        rank_order[Rank.ACE] = 1
        if not straight and cls.__check_straights(ranks, rank_order):
            straight = True

        # flush
        flush = False
        if any(count >= 5 for count in suit_counts.values()):
            flush = True

        if flush and all(
            card in ranks
            for card in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        ):
            return cls.ROYAL_FLUSH
        elif flush and straight:
            return cls.STRAIGHT_FLUSH
        elif straight:
            return cls.STRAIGHT
        elif flush:
            return cls.FLUSH
        # full house: one three-of-a-kind and one pair
        elif all(count in rank_counts.values() for count in [2, 3]):
            return cls.FULL_HOUSE
        # three-of-a-kind
        elif 3 in rank_counts.values():
            return cls.THREE_OF_A_KIND
        # Check for two pair (only if not full house)
        elif list(rank_counts.values()).count(2) >= 2:
            return cls.TWO_PAIR
        elif 2 in rank_counts.values():
            return cls.PAIR
        else:
            return cls.HIGH_CARD

    def base_scoring(self) -> Scoring:
        match self:
            case Hand.HIGH_CARD:
                return Scoring(chips=5, mult=1)
            case Hand.PAIR:
                return Scoring(chips=10, mult=2)
            case Hand.TWO_PAIR:
                return Scoring(chips=20, mult=2)
            case Hand.THREE_OF_A_KIND:
                return Scoring(chips=30, mult=3)
            case Hand.STRAIGHT:
                return Scoring(chips=30, mult=4)
            case Hand.FLUSH:
                return Scoring(chips=35, mult=4)
            case Hand.FULL_HOUSE:
                return Scoring(chips=40, mult=4)
            case Hand.FOUR_OF_A_KIND:
                return Scoring(chips=60, mult=7)
            case Hand.STRAIGHT_FLUSH:
                return Scoring(chips=100, mult=8)
            case Hand.ROYAL_FLUSH:
                return Scoring(chips=100, mult=8)


class DumbBalatro:
    deck: list[Card]

    def __init__(self) -> None:
        self.deck = [Card(suit, rank) for suit in Suit for rank in Rank]


if __name__ == "__main__":
    balatro = DumbBalatro()
    print(balatro.deck)
