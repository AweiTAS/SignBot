import numpy as np
from .RoomLine import RoomLine
from .RoomLine import ArrowDir

def min_edit_distance(a, b):
        dp = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
        for i in range(len(a) + 1):
            dp[i][0] = i
        for j in range(len(b) + 1):
            dp[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[-1][-1]

class TextProcessor:

    currentFloor = 0
    currentBuilding = 'A'
    savedLines = []

    def __init__(self) -> None:
        self.savedLines = []
        return
    
    def __roomPattern(self, text):
        rl = RoomLine()
        text = text.replace(" ","")
        text = text.replace(".","")
        if(min_edit_distance(text[0:5], 'Rooms') > 2):
            return None
        text = text[5:]
        if(len(text) <= 7):
            try:
                self.currentFloor = int(text[0])
            except:
                pass

            try:
                self.currentFloor = text[1]
            except:
                pass
            text = text[2:]
            try:
                rl.left = int(text)
                rl.right = int(text)
                return rl
            except:
                return None
        else:
            if("-" in text):
                i = text.find("-")
                left = text[:i]
                right = text[i+1:]
            elif("&" in text):
                i = text.find("&")
                left = text[:i]
                right = text[i+1:]
            else:
                left = text
                right = text
            if(min_edit_distance(right[0:5], 'Rooms') <= 2):
                right = right[5:]
            if(left[0]==right[0]=='G'):
                self.currentFloor = "G"
            elif(left[0]==right[0]):
                self.currentFloor = int(left[0])

            if(left[1]==right[1]=='A'):
                self.currentBuilding = 'A'
            elif(left[1]==right[1]=='B'):
                self.currentBuilding = 'B'
            left = left[2:]
            right = right[2:]
            try:
                rl.left = int(left[:3])
                rl.right = int(right[:3])
            except:
                return None
            
            right = right[4:]
            if(len(right)>=5):
                print(right)
                srl = self.__singleRoomPattern(right)
                if srl is not None:
                    self.savedLines.append(srl)

            return rl

    def __singleRoomPattern(self, text):
        rl = RoomLine()
        text = text.replace(" ","")
        text = text.replace(".","")
        if((not len(text)==5) or (text[1]!='A' and text[1]!='B')):
            return None
        try:
            self.currentFloor = int(text[0])
            self.currentBuilding = text[1]
            text = text[2:]
            rl.left = int(text)
            rl.right = int(text)
        except:
                return None
        return rl

    def addLine(self, text, arrowDir):
        rl = self.__roomPattern(text)
        if rl is not None:
            rl.arrowDir = arrowDir.argmax()
            self.savedLines.append(rl)
            return
        
        srl = self.__singleRoomPattern(text)
        if srl is not None:
            self.savedLines.append(srl)
        return
    
    def addLineWithoutArr(self, text):
        rl = self.__roomPattern(text)
        if rl is not None:
            rl.arrowDir = 0
            self.savedLines.append(rl)
            return
        srl = self.__singleRoomPattern(text)
        if srl is not None:
            self.savedLines.append(srl)
        return
    

    def clearLines(self):
        self.savedLines = []
        return
    

    def searchForLoc(self, target):
        for rl in self.savedLines:
            if rl.left <= target and rl.right >= target:
                return rl
        return None
    
    def __str__(self) -> str:
        ret = ''
        for line in self.savedLines:
            ret = ret + '\n' + str(line)
        return ret
    
if __name__ == "__main__":
    try:
        print(int('1a3'))
    except ValueError:
        pass