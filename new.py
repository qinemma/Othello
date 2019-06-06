import time, math, copy
INFINITY = float("inf")

class AI(object):
    def __init__(self, game):
        self.game = game
        self.move = (-1, -1)
        self.timeLimit = 3  # 3 seconds is the time limit for search
        self.debug = False  # True for debugging

    def performMove(self):
        # Iterative Deepening MiniMax Search with Alpha-Beta Pruning
        tmpBoard = [row[:] for row in self.game.board]  # we don't want to make changes to the game board

        #change to other algorithm here
        self.move = self.miniMax(tmpBoard)
        self.move = self.negaScout(tmpBoard)

        # perform move (there must be an available move)
        self.game.performMove(self.move[0], self.move[1])

    def miniMax(self, board):
        startTime = time.time()
        timeElapsed = 0
        depth = 0
        optimalMove = (-1, -1)
        optimalBoard = board
        stopDigging = False

        while not stopDigging and timeElapsed < self.timeLimit:
            stopDigging, optimalBoard = self.miniMaxHelper(board, depth, 3, -INFINITY, INFINITY, 2)
            endTime = time.time()
            timeElapsed += endTime - startTime
            startTime = endTime
            depth += 1
        print("[Console MSG] Time used by AI: " + str(timeElapsed))

        for row in range(0, 8):
            for col in range(0, 8):
                if board[row][col] != optimalBoard[row][col]:
                    optimalMove = (row, col)

        return optimalMove


    def miniMaxHelper(self, board, depth, maxDepth, alpha, beta, player):
        if self.debug:
            print("Level: " + str(depth))
        stopDigging = False

        if (not self.game.moveCanBeMade(board, player) or depth == maxDepth):
            return stopDigging, board

        successorBoards = self.findSuccessorBoards(board, player)
        if len(successorBoards) == 0:
            stopDigging = True
            return (stopDigging, board)
        bestBoard = None

        if player == 2:
            maxVal = -INFINITY
            for successor in successorBoards:
                stopDigging, miniBoard = self.miniMaxHelper(successor, depth, alpha, beta, 1)
                utility = self.utilityOf(miniBoard)
                if utility > maxVal:
                    bestBoard = successor
                alpha = max(alpha, utility)
                if utility >= beta:
                    return stopDigging, successor

        else:
            minVal = INFINITY
            for successor in successorBoards:
                stopDigging, maxBoard = self.self.miniMaxHelper(successor, depth, alpha, beta, 2)
                utility = self.utilityOf(maxBoard)
                if utility < minVal:
                    bestBoard = successor
                beta = min(beta, utility)
                if utility <= alpha:
                    return stopDigging, successor

        return stopDigging, bestBoard

    def negaScout(self, board):
        optimalFlipping = 0
        if board[0][0] == 0:
            flippingAtCorner = self.game.placePiece(board, 0, 0, 2, PLAYMODE=False)
            if flippingAtCorner > optimalFlipping:
                optimalFlipping = flippingAtCorner
                optimalMove = (0, 0)
        if board[7][0] == 0:
            flippingAtCorner = self.game.placePiece(board, 7, 0, 2, PLAYMODE=False)
            if flippingAtCorner > optimalFlipping:
                optimalFlipping = flippingAtCorner
                optimalMove = (7, 0)
        if board[0][7] == 0:
            flippingAtCorner = self.game.placePiece(board, 0, 7, 2, PLAYMODE=False)
            if flippingAtCorner > optimalFlipping:
                optimalFlipping = flippingAtCorner
                optimalMove = (0, 7)
        if board[7][7] == 0:
            flippingAtCorner = self.game.placePiece(board, 7, 7, 2, PLAYMODE=False)
            if flippingAtCorner > optimalFlipping:
                optimalFlipping = flippingAtCorner
                optimalMove = (7, 7)
        if optimalFlipping > 0:
            return optimalMove

        startTime = time.time()
        timeElapsed = 0
        depth = 4
        optimalMove = (-1, -1)
        optimalBoard = board
        stopDigging = False
        while not stopDigging and timeElapsed < self.timeLimit:
            (stopDigging, optimalBoard, alpha) = self.negaScoutHelper(board, 2, 0, depth, -INFINITY, INFINITY, 1)
            endTime = time.time()
            timeElapsed += endTime - startTime
            startTime = endTime
            depth += 1
        print("[Console MSG] Time used by AI: " + str(timeElapsed))

        for row in range(0, 8):
            for col in range(0, 8):
                if board[row][col] != optimalBoard[row][col]:
                    optimalMove = (row, col)

        return optimalMove

    def negaScoutHelper(self, board, player, depth, maxDepth, alpha, beta, color):
        if self.debug:
            print("here")
        stopDigging = False

        if (not self.game.moveCanBeMade(board, player) or depth == maxDepth):
            utility = self.utilityOf(board)
            return stopDigging, board, color * utility

        successorBoards = self.findSuccessorBoards(board, player)
        '''
            if len(successorBoards) == 0:
                stopDigging = True
                return stopDigging, board, 0
        '''
        bestBoard = None
        first = True
        for successor in successorBoards:
            if first:
                score = -self.negaScoutHelper(successor, player, depth + 1, maxDepth, -beta, -alpha, -color)[2]
                first = False
            else:
                score = -self.negaScoutHelper(successor, player, depth + 1, maxDepth, -alpha - 1, -alpha, -color)[2]
                if alpha < score < beta:
                    score = -self.negaScoutHelper(successor, player, depth + 1, maxDepth, -beta, -score, -color)[2]

            if score >= alpha:
                bestBoard = successor
                alpha = score

            if alpha >= beta:
                break

        return stopDigging, bestBoard, alpha

    # return a list of successor boards
    def findSuccessorBoards(self, board, player):
        successorBoards = []
        for row in range(0, 8):
            for col in range(0, 8):
                if board[row][col] == 0:
                    numAvailableMoves = self.game.placePiece(board, row, col, player, PLAYMODE=False)
                    if numAvailableMoves > 0:
                        successorBoard = copy.deepcopy([row[:] for row in board])
                        successorBoard[row][col] = player
                        successorBoards.append(successorBoard)
        return successorBoards


    def utilityOf(self, board):

        return 0


