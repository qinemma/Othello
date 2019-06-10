import time, math, copy
import numpy as np
import pandas as pd
import machineLearning
import pickle
INFINITY = float("inf")



class GameAI(object):
	def __init__(self, game):
		super().__init__()
		self.game = game
		self.move = (-1,-1)
		self.timeLimit = 3  # 3 seconds is the time limit for search
		self.debug = False  # True for debugging
		self.fileObject = open("decisionTree", 'rb')
		self.tree = pickle.load(self.fileObject)

	# AI perform move (there must be an available move due to the pre-move check)
	def performMove(self, index):
		# Iterative Deepening MiniMax Search with Alpha-Beta Pruning
		tmpBoard = [row[:] for row in self.game.board] # we don't want to make changes to the game board
		if index == 0:
			self.move = self.miniMax(tmpBoard)
			print("minimax")
			print(self.move)
		else:
			self.move = self.negaScout(tmpBoard)
			print("negascout")

			#testing decision tree
			#self.move = self.oriminiMax(tmpBoard)
			#print("oriMinimax")
			print(self.move)

		if self.move is None:
			#print("here")
			return
		else:
			# perform move (there must be an available move)
			self.game.performMove(self.move[0], self.move[1])

	def getSortedNode(self, board, player):
		sortedNodes = []
		successorBoards = self.findSuccessorBoards(board, player)
		for successorBoard in successorBoards:
			sortedNodes.append((successorBoard, self.utilityOf(successorBoard, player)))
		sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=True)
		sortedNodes = [node[0] for node in sortedNodes]

		return sortedNodes

	""" Iterative Deepening MiniMax Search Algorithm within Time Limit
		From depth = 3, if still within the time limit, continue search to get more insight.
		Return the optimal move within limited resources. 
	"""
	def miniMax(self, board):

		print("here")
		startTime = time.time()
		timeElapsed = 0
		depth = 3
		optimalMove = (-1, -1)
		optimalBoard = board
		stopDigging = False
		while not stopDigging and timeElapsed < self.timeLimit:
			stopDigging, optimalBoard = self.IDMiniMax(board, 0, depth, 2, -INFINITY, INFINITY)
			endTime = time.time()
			timeElapsed += endTime - startTime
			startTime = endTime
			depth += 1
		print("[Console MSG] Time used by AI: " + str(timeElapsed))

		if optimalBoard == board:
			return None

		for row in range(0, 8):
			for col in range(0, 8):
				if board[row][col] != optimalBoard[row][col]:
					optimalMove = (row, col)

		print(np.asarray(optimalBoard).reshape(8, 8))
		return optimalMove

	""" Iterative Deepening MiniMax Search with Alpha-Beta Pruning
		board - state at current node
		player - player at current node (AI - white - maximizer; Player - black - minimizer)
		currentLevel - level at current node
		maxLevel - used to judge whether go deeper or not
		Return the optimal board (state) found in the current level for the current node.
	"""
	def IDMiniMax(self, board, currentLevel, maxLevel, player, alpha, beta):
		if self.debug:
			print("Level: " + str(currentLevel) + " maxLevel: " + str(maxLevel))
		stopDigging = False
		if (not self.game.moveCanBeMade(board, player) or currentLevel == maxLevel):
			return (stopDigging, board)

		successorBoards = self.findSuccessorBoards(board, player)
		if len(successorBoards) == 0:
			stopDigging = True
			return stopDigging, board
		bestBoard = None

		if player == 2:
			maxValue = -INFINITY
			for successor in successorBoards:
				stopDigging, lookaheadBoard = self.IDMiniMax(successor, currentLevel+1, maxLevel, 1, alpha, beta)
				utility = self.utilityOf(lookaheadBoard, player)
				if utility > maxValue:
					maxValue = utility
					bestBoard = successor
				alpha = max(alpha, utility)
				if utility >= beta:
					#print("alphaBeta is pruning", successor)
					return stopDigging, successor # prune
		else:
			minValue = INFINITY
			for successor in successorBoards:
				stopDigging, lookaheadBoard = self.IDMiniMax(successor, currentLevel+1, maxLevel, 2, alpha, beta)
				utility = self.utilityOf(lookaheadBoard, player)
				if utility < minValue:
					minValue = utility
					bestBoard = successor
				beta = min(beta, utility)
				if utility <= alpha:
					#print("alphaBeta is pruning", successor)
					return stopDigging, successor  # prune

		return stopDigging, bestBoard

	def negaScout(self, board):

		startTime = time.time()
		timeElapsed = 0
		depth = 3
		optimalMove = (-1, -1)
		optimalBoard = board
		stopDigging = False
		while not stopDigging and timeElapsed < self.timeLimit:
			# (stopDigging, optimalBoard, alpha) = self.negaScoutHelper(board, 2, depth, -INFINITY, INFINITY, 1)
			maxScore = -INFINITY
			for successor in self.getSortedNode(board, 1):
				point = self.negaScoutHelper2(successor, 1, depth, -INFINITY, INFINITY, 1)
				if point > maxScore:
					maxScore = point
					optimalBoard = successor
			endTime = time.time()
			timeElapsed += endTime - startTime
			startTime = endTime
			depth += 1
		print("[Console MSG] Time used by AI: " + str(timeElapsed))

		if optimalBoard == board:
			print("here")
			return None

		for row in range(0, 8):
			for col in range(0, 8):
				if board[row][col] != optimalBoard[row][col]:
					optimalMove = (row, col)

		print(np.asarray(optimalBoard).reshape(8, 8))

		print(optimalMove)

		return optimalMove

	def negaScoutHelper2(self, board, player, depth, alpha, beta, color):

		if not self.game.moveCanBeMade(board, player) or depth == 0:
			return self.utilityOf(board, player) * color
		successorBoards = self.getSortedNode(board, player)
		first = True

		for successor in successorBoards:
			if not first:
				score = -self.negaScoutHelper2(successor, player, depth - 1, -alpha - 1, -alpha, -color)
				if alpha < score < beta:
					score = -self.negaScoutHelper2(successor, player, depth - 1, -beta, -score, -color)
			else:
				first = False
				score = -self.negaScoutHelper2(successor, player, depth - 1, -beta, -alpha, -color)

			alpha = max(alpha, score)
			if alpha >= beta:
				#print("negascout is pruning", successor)
				break

		return alpha


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

	# evaluation function (heuristics for non-final node) in this state (board)
	def utilityOf(self, board, player):
		board_mobility = self.mobility(board, player)
		board_frontier = self.frontierSquares(board, player)
		board_corners = self.corners(board, player)
		xsquares, csquares = self.x_c_squares(board, player)
		board_parity = self.parity(board)
		board_state = self.gameState(board)
		df = pd.Series([board_mobility, board_frontier, board_corners, xsquares, csquares, board_parity, board_state],
					   index=["numMoves", "frontier", "corners", "Xsquares", "CSquares", "parity", "state"])

		return machineLearning.predict(df, self.tree)

	# mobility, number of moves a player can make minus number of moves its opponent can make
	def mobility(self, board, player):

		blackMovesFound = self.findSuccessorBoards(board, 1)
		whiteMovesFound = self.findSuccessorBoards(board, 2)
		if player == 1:
			return len(blackMovesFound) - len(whiteMovesFound)
		elif player == 2:
			return len(whiteMovesFound) - len(blackMovesFound)
		else:
			return 0

	# number of frontier that player occupies
	def frontierSquares(self, board, player):
		if player == 1:
			opp = 2
		if player == 2:
			opp = 1
		coords_x, coords_y = np.where(np.array(board) == player)  # coordinates that surround opponents' pieces
		opp_coords_x, opp_coords_y = np.where(np.array(board) == opp)
		frontier = []
		frontier_opp = []
		sur_player = []
		for i in range(len(coords_x)):
			for row in [-1, 0, 1]:
				for col in [-1, 0, 1]:
					x = coords_x[i] + row
					y = coords_y[i] + col
					if 0 <= x < 8 and 0 <= y < 8:
						np.append(sur_player, np.array([x, y]))
			if len(sur_player) > 0:
				sur_player = np.unique(np.asarray(sur_player), axis=0)
				for i in range(len(sur_player)):
					if board[sur_player[i][0]][sur_player[i][1]] == 0:
						np.append(frontier, sur_player[i])


		sur_opp = []
		for i in range(len(opp_coords_x)):
			for row in [-1, 0, 1]:
				for col in [-1, 0, 1]:
					x = opp_coords_x[i] + row
					y = opp_coords_y[i] + col
					if 0 <= x < 8 and 0 <= y < 8:
						#sur_opp.append(np.array([x, y]))
						np.append(sur_opp, np.array([x, y]))
			if len(sur_opp) > 0:
				sur_opp = np.unique(np.asarray(sur_opp), axis=0)
				for i in range(len(sur_opp)):
					if board[sur_opp[i][0]][sur_opp[i][1]] == 0:
						np.append(frontier_opp, sur_opp[i])

		return len(frontier) - len(frontier_opp)


	#number of corners the player occupies
	def corners(self, board, player):
		corners = np.array([[0, 0], [0, 7], [7, 0], [7, 7]])
		if player == 1:
			opp = 2
		if player == 2:
			opp = 1
		black_corner = 0
		white_corner = 0
		for corner in corners:
			if board[corner[0]][corner[1]] == 0:
				continue
			elif board[corner[0]][corner[1]] == 1:
				black_corner += 1
			else:
				white_corner += 1

		if player == 1:
			return black_corner - white_corner
		elif player == 2:
			return white_corner - black_corner
		else:
			return 0  # bit different from how the data is created, does not matter, because player 0 gets subsetted

	#number of x_c squares player occupies
	def x_c_squares(self, board, player):
		corners = np.array([[0, 0], [0, 7], [7, 0], [7, 7]])
		x_squares = np.array([[1, 1], [1, 6], [6, 1], [6, 6]])
		c_squares1 = np.array([[0, 1], [1, 7], [6, 0], [7, 6]])
		c_squares2 = np.array([[1, 0], [0, 6], [7, 1], [6, 7]])
		if player == 1:
			opp = 2
		if player == 2:
			opp = 1
		player_x_squares = 0
		opp_x_squares = 0
		player_c_squares = 0
		opp_c_squares = 0
		for i in range(len(x_squares)):
			if board[corners[i][0]][corners[i][1]] == 0:
				if board[x_squares[i][0]][x_squares[i][1]] == player:
					player_x_squares += 1
				if board[c_squares1[i][0]][c_squares1[i][1]] == player:
					player_c_squares += 1
				if board[c_squares2[i][0]][c_squares2[i][1]] == player:
					player_c_squares += 1
				if board[x_squares[i][0]][x_squares[i][1]] == opp:
					opp_x_squares += 1
				if board[c_squares1[i][0]][c_squares1[i][1]] == opp:
					opp_c_squares += 1
				if board[c_squares2[i][0]][c_squares2[i][1]] == opp:
					opp_c_squares += 1
				else:
					continue
		XSquares = player_x_squares - opp_x_squares
		CSquares = player_c_squares - opp_c_squares
		return XSquares, CSquares

	def parity(self, board):
		progress = 0
		for row in range(8):
			for col in range(8):
				if board[row][col] != 0:
					progress += 1


		if progress % 2 == 0:
			parity = 0
		else:
			parity = 1
		return parity

	#which game state the player is on
	def gameState(self, board):
		progress = 0
		for row in range(8):
			for col in range(8):
				if board[row][col] != 0:
					progress += 1

		if progress % 61 <= 20:
			return "beginning"
		elif progress % 61 <= 40:
			return "middle"
		else:
			return "end"

	#Code later is used to test the how well the decision performs
	#Original code from the ai.py

	def oriminiMax(self, board):
		startTime = time.time()
		timeElapsed = 0
		depth = 2
		optimalMove = (-1, -1)
		optimalBoard = board
		stopDigging = False
		while not stopDigging and timeElapsed < self.timeLimit:
			(stopDigging, optimalBoard) = self.IDMiniMax(board, 0, depth, 1, -INFINITY, INFINITY)
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

	""" Iterative Deepening MiniMax Search with Alpha-Beta Pruning
        board - state at current node
        player - player at current node (AI - white - maximizer; Player - black - minimizer)
        currentLevel - level at current node
        maxLevel - used to judge whether go deeper or not
        Return the optimal board (state) found in the current level for the current node.
    """

	def oriIDMiniMax(self, board, currentLevel, maxLevel, player, alpha, beta):
		if self.debug:
			print("Level: " + str(currentLevel) + " maxLevel: " + str(maxLevel))
		stopDigging = False
		if (not self.game.moveCanBeMade(board, player) or currentLevel == maxLevel):
			return (stopDigging, board)

		successorBoards = self.findSuccessorBoards(board, player)
		if len(successorBoards) == 0:
			stopDigging = True
			return (stopDigging, board)
		bestBoard = None

		if player == 2:
			maxValue = -INFINITY
			for idx in range(0, len(successorBoards)):
				stopDigging, lookaheadBoard = self.oriIDMiniMax(successorBoards[idx], currentLevel + 1, maxLevel, 1, alpha,
															 beta)
				utility = self.oriUtilityOf(lookaheadBoard)
				if utility > maxValue:
					maxValue = utility
					bestBoard = successorBoards[idx]
				alpha = max(alpha, utility)
				if utility >= beta:
					return (stopDigging, successorBoards[idx])  # prune
		else:
			minValue = INFINITY
			for idx in range(0, len(successorBoards)):
				stopDigging, lookaheadBoard = self.oriIDMiniMax(successorBoards[idx], currentLevel + 1, maxLevel, 2, alpha,
															 beta)
				utility = self.oriUtilityOf(lookaheadBoard)
				if utility < minValue:
					minValue = utility
					bestBoard = successorBoards[idx]
				beta = min(beta, utility)
				if utility <= alpha:
					return (stopDigging, successorBoards[idx])  # prune

		return (stopDigging, bestBoard)


	def oriUtilityOf(self, board):
		return self.oriPieceDifference(board) + self.oriCornerCaptions(board) + self.oriCornerCloseness(board) + self.oriMobility(board) + self.oriStability(board)

	# piece difference when evaluating 
	def oriPieceDifference(self, board):
		allTiles = [item for sublist in board for item in sublist]
		whiteTiles = sum(1 for tile in allTiles if tile == 2)
		blackTiles = sum(1 for tile in allTiles if tile == 1)

		if whiteTiles > blackTiles:
			return (whiteTiles / (blackTiles + whiteTiles)) * 100
		else:
			return - (blackTiles / (blackTiles + whiteTiles)) * 100

	# how many corners are owned by each player
	def oriCornerCaptions(self, board):
		numCorners = [0, 0]
		if board[0][0] == 1:
			numCorners[0] += 1
		else:
			numCorners[1] += 1
		if board[0][7] == 1:
			numCorners[0] += 1
		else:
			numCorners[1] += 1
		if board[7][0] == 1:
			numCorners[0] += 1
		else:
			numCorners[1] += 1
		if board[7][7] == 1:
			numCorners[0] += 1
		else:
			numCorners[1] += 1

		return 50 * (numCorners[1] - numCorners[0])

	# how many corner-closeness pieces are owned by each player
	def oriCornerCloseness(self, board):
		numCorners = [0, 0]
		for row in range(1, 7):
			if board[row][0] == 1:
				numCorners[0] += 1
			elif board[row][0] == 2:
				numCorners[1] += 1

			if board[row][7] == 1:
				numCorners[0] += 1
			elif board[row][7] == 2:
				numCorners[1] += 1

		for col in range(1, 7):
			if board[0][col] == 1:
				numCorners[0] += 1
			elif board[7][col] == 2:
				numCorners[1] += 1

			if board[row][7] == 1:
				numCorners[0] += 1
			elif board[row][7] == 2:
				numCorners[1] += 1		

		return 4 * (numCorners[1] - numCorners[0])

	# relative mobility of a player to another (how many steps can a player move)
	def oriMobility(self, board):
		blackMobility = self.game.moveCanBeMade(board, 1)
		whiteMobility = self.game.moveCanBeMade(board, 2)

		if blackMobility + whiteMobility == 0:
			return 0
		else:
			return 100 * whiteMobility / (whiteMobility + blackMobility)

	# for a piece: stable - 1; semi-stable: 0; instable - -1
	def oriStability(self, board):
		stability = [0, 0]
		blackStability, whiteStability = stability[0], stability[1]

		for row in range(1, 7):
			for col in range(1, 7):
				instabilityScale = 0
				current = board[row][col]
				if current == 0:
					continue
				if board[row+1][col+1] == 0:
					instabilityScale += 1
				if board[row-1][col-1] == 0:
					instabilityScale += 1
				if board[row+1][col] == 0:
					instabilityScale += 1
				if board[row-1][col] == 0:
					instabilityScale += 1
				if board[row+1][col-1] == 0:
					instabilityScale += 1
				if board[row-1][col+1] == 0:
					instabilityScale += 1
				if board[row][col+1] == 0:
					instabilityScale += 1
				if board[row][col-1] == 0:
					instabilityScale += 1

				if instabilityScale >= 7:
					stability[current - 1] -= 1;
				elif instabilityScale <= 3:
					stability[current - 1] += 1;

		whiteStability, blackStability = stability[1], stability[0]

		if whiteStability + blackStability == 0:
			return 0
		else:
			return 100 * whiteStability / (whiteStability + blackStability)
	





