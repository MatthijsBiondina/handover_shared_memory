from enum import IntEnum


class States(IntEnum):
    RESTING = 0
    APPROACHING = 1
    REACHING = 2
    GRASPING = 3
    RETRACT = 4
    GIVE_BACK = 5

    def __str__(self):
        return self.name
