from enum import Enum
class ArrowDir(Enum):
    unknown = 0
    right = 1
    top = 2
    left = 3


class RoomLine:

    left = -1
    right = -1
    arrowDir = 0

    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        return str("left:") + str(self.left) + str(" right:") + str(self.right) + str(" arrow:") + str(self.arrowDir)

class SingleRoomLine:

    room = -1

    def __init__(self) -> None:
        pass