import sys
import os

sys.path.append(os.path.realpath("."))

import unittest

from dumb_balatro import DumbBalatro, Card, Hand, HandType, Rank, Suit


class TestDumbBalatro(unittest.TestCase):
    def test_straight_flush(self):
        cards = [
            Card(Suit.SPADES, Rank.NINE),
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.STRAIGHT_FLUSH)
        self.assertEqual(set(hand.indices), set(range(len(hand.indices))))

    def test_straight_ace_high(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.ACE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.STRAIGHT)
        self.assertEqual(set(hand.indices), set(range(len(hand.indices))))

    def test_straight_ace_low(self):
        cards = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.SPADES, Rank.FIVE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.STRAIGHT)
        self.assertEqual(set(hand.indices), set(range(len(hand.indices))))

    def test_flush(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.ACE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.FLUSH)
        self.assertEqual(set(hand.indices), set(range(len(hand.indices))))

    def test_full_house(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.HEARTS, Rank.ACE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.FULL_HOUSE)
        self.assertEqual(set(hand.indices), set(range(len(hand.indices))))

    def test_three_of_a_kind(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.DIAMONDS, Rank.FIVE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.THREE_OF_A_KIND)
        self.assertEqual(hand.indices, list(range(len(hand.indices))))

    def test_two_pair(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.CLUBS, Rank.TWO),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.TWO_PAIR)
        self.assertEqual(hand.indices, [0, 1, 2, 3])

    def test_pair(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.CLUBS, Rank.TWO),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.PAIR)
        self.assertEqual(hand.indices, [0, 1])

    def test_high_card(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        hand = Hand.classify_hand(cards)
        self.assertEqual(hand.hand_type, HandType.HIGH_CARD)
        self.assertEqual(hand.indices, [0])

    def test_game(self):
        balatro = DumbBalatro(seed=1234)
        self.assertEqual(balatro.play([0, 2, 4, 5, 7], discard=True), None)
        self.assertEqual(balatro.discards, 3)
        self.assertEqual(balatro.play([3, 4, 5, 7]), 15)
        self.assertEqual(balatro.hands, 3)
        self.assertEqual(balatro.play([0, 1, 2, 3, 4]), 284)
        self.assertEqual(balatro.hands, 2)


if __name__ == "__main__":
    unittest.main()
