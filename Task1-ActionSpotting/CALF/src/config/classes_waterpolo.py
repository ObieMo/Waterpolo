import torch


# Event name to label index for water polo
EVENT_DICTIONARY_V2 = {
    "GOAL": 0,
    "MissedShot": 1,
}

INVERSE_EVENT_DICTIONARY_V2 = {
    0: "GOAL",
    1: "MissedShot",
}

# K parameters (seconds) for 2 classes.
# Col 0 reuses SoccerNet "Goal" profile.
# Col 1 reuses SoccerNet "Shots off target" profile for MissedShot.
K_V2 = torch.FloatTensor(
    [
        [-16, -16],
        [-8, -8],
        [8, 8],
        [16, 16],
    ]
)
