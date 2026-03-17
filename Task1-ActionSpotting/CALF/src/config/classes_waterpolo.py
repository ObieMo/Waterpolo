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
        [-20, -8],
        [-10, -4],
        [60, 4],
        [90, 8],
    ]
)

if torch.cuda.is_available():
    K_V2 = K_V2.cuda()
