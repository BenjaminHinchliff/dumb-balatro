from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary

from dumb_balatro.game import Card, DumbBalatro, Rank, Suit

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

        self.action_space = Dict(
            {
                "cards": Box(
                    low=0, high=1, shape=(MAX_CARDS_PER_HAND, N_CARDS), dtype=np.int_
                ),  # 5x52 matrix for card selection
                "play_or_discard": Box(
                    low=0, high=1, shape=(MAX_CARDS_PER_HAND,), dtype=np.int_
                ),  # 5x1 binary vector (1 = play, 0 = discard)
            }
        )

    # with the extra complexity this brings I'm starting to think it *might* be stupid
    def card_to_int(self, suit: Suit, rank: Rank):
        return list(Suit).index(suit) * len(Rank) + list(Rank).index(rank)

    def int_to_card(self, n: int) -> tuple[str, str]:
        return list(Suit)[n // len(Suit)], list(Rank)[n % len(Rank)]

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

    def build_choices_from_action(self, actions, hand_encoded):
        card_choices = torch.multinomial(
            (
                actions[:, : N_CARDS * MAX_CARDS_PER_HAND]
                .view(-1, MAX_CARDS_PER_HAND, N_CARDS)
                .softmax(dim=2)
                * hand_encoded.unsqueeze(1)
            ).view(-1, N_CARDS),
            num_samples=1,
        )
        card_choices = card_choices.view(-1, MAX_CARDS_PER_HAND)
        option_choice = actions[:, N_CARDS * MAX_CARDS_PER_HAND :].max(1).indices
        return torch.cat((card_choices, option_choice.unsqueeze(1)), dim=1)

    # attempt to generously convert what the model "wants" based on the actual hand
    def action_to_command(self, tensor) -> tuple[list, bool]:
        hand = self.balatro.hand
        action = tensor
        cards = action[:-1]
        option = action[-1]
        selection = set()
        for card in cards:
            suit, rank = self.int_to_card(int(card.item()))
            try:
                hand_index = next(
                    (i for i, c in enumerate(hand) if c.suit == suit and c.rank == rank)
                )
                # lua indexes start at 1 (guess how I found out)
                selection.add(hand_index + 1)
            except StopIteration:
                pass

        return (list(selection), option)

    def step(
        self, action: object
    ) -> tuple[object, SupportsFloat, bool, bool, dict[str, Any]]:
        super().step(action)
        action = torch.from_numpy(action)

        choices = self.build_choices_from_action(action, self._enc_hand())
        indices, option = self.action_to_command(choices)

        reward = self.balatro.play(indices, discard=option) or 0
        observation = self._get_obs()
        terminated = self.balatro.is_ended()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="DumbBalatro",
    entry_point=DumbBalatroGym,
)
