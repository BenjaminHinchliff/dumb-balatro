from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary

from dumb_balatro.game import Card, DumbBalatro, InvalidPlay, Rank, Suit

import torch
import torch.nn.functional as F
import numpy as np

N_CARDS = 52
MAX_CARDS_PER_HAND = 5


class DumbBalatroGym(gym.Env):
    balatro: DumbBalatro

    def __init__(self) -> None:
        super().__init__()

        self.balatro = DumbBalatro()
        self.observation_space = Dict(
            {
                "hand": MultiBinary(
                    N_CARDS
                ),  # Multi-hot encoding of the 8 cards in hand
            }
        )

        self.action_space = Box(
            low=0,
            high=1,
            shape=(MAX_CARDS_PER_HAND, N_CARDS + 1),
            dtype=np.float32,
        )  # 5x52 matrix for card selection, plus one for whether to play or discard (and yes it's stupid but necessary :/)

    # with the extra complexity this brings I'm starting to think it *might* be stupid
    def card_to_int(self, suit: Suit, rank: Rank):
        return list(Suit).index(suit) * len(Rank) + list(Rank).index(rank)

    def int_to_card(self, n: int) -> tuple[str, str]:
        return list(Suit)[n // len(Rank)], list(Rank)[n % len(Rank)]

    def hand_to_ints(self, hand: list[Card]) -> list[int]:
        return [self.card_to_int(card.suit, card.rank) for card in hand]

    def _enc_hand(self):
        hand = self.hand_to_ints(self.balatro.hand)
        # currently the state is just the state of the hand (multi-hot encoded)
        return F.one_hot(torch.tensor(hand), num_classes=len(Suit) * len(Rank)).sum(
            dim=0
        )

    def _get_obs(self) -> dict[str, Any]:
        return {"hand": self._enc_hand().numpy()}

    def _get_info(self):
        return {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[object, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.balatro.reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def build_choices_from_action(self, action, hand_encoded):
        card_choices = torch.multinomial(
            action.softmax(dim=1) * hand_encoded,
            num_samples=1,
        )
        return card_choices

    # attempt to generously convert what the model "wants" based on the actual hand
    def action_to_command(self, cards) -> list:
        hand = self.balatro.hand
        selection = set()
        for card in cards:
            suit, rank = self.int_to_card(int(card.item()))
            try:
                hand_index = next(
                    (i for i, c in enumerate(hand) if c.suit == suit and c.rank == rank)
                )
                selection.add(hand_index)
            except StopIteration:
                pass

        return list(selection)

    def step(
        self, action: dict
    ) -> tuple[object, SupportsFloat, bool, bool, dict[str, Any]]:
        cards = torch.from_numpy(action[:, :-1])
        play_or_discard = action[0, -1] > 0.5

        choices = self.build_choices_from_action(cards, self._enc_hand())
        indices = self.action_to_command(choices)

        try:
            reward = self.balatro.play(indices, discard=play_or_discard) or 0
        except InvalidPlay as e:
            reward = -1
        observation = self._get_obs()
        terminated = self.balatro.is_ended()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="DumbBalatro",
    entry_point=DumbBalatroGym,
)
