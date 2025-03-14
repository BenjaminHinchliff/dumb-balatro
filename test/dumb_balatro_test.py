import sys
import os

sys.path.append(os.path.realpath("."))

import unittest

from dumb_balatro import Card, Hand, Rank, Suit


class TestDumbBalatro(unittest.TestCase):
    def test_royal_flush(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.ACE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.ROYAL_FLUSH)

    def test_straight_flush(self):
        cards = [
            Card(Suit.SPADES, Rank.NINE),
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.STRAIGHT_FLUSH)

    def test_straight_ace_high(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.ACE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.STRAIGHT)

    def test_straight_ace_low(self):
        cards = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.SPADES, Rank.FIVE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.STRAIGHT)

    def test_flush(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.ACE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.FLUSH)

    def test_full_house(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.HEARTS, Rank.ACE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.FULL_HOUSE)

    def test_three_of_a_kind(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.TEN),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.THREE_OF_A_KIND)

    def test_two_pair(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.EIGHT),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.TWO_PAIR)

    def test_pair(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.TEN),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.PAIR)

    def test_high_card(self):
        cards = [
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        self.assertEqual(Hand.classify_hand(cards), Hand.HIGH_CARD)


if __name__ == "__main__":
    unittest.main()
