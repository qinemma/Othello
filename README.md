# Othello
credit to James.Qiu by email: jamesqiu@hku.hk in github

## An Screenshot of Game GUI

![](GUI.png)

## Architecture

### Component One: Game Engine - main.py

Game engine handles all I/O events, repaints the board and manages games.

### Component Two: Game Logic Implementation

All game logic will be handled by othello.py.

### Component Three: Artificial Intelligence

Artificial intelligence algorithm will be implemented in this component.

## How to Run?

In Windows,

- Install python 3 (3.6.1 or above)
- Install pygame `py -m pip install pygame`
- Go to the root directory and run `py main.py`

In Linux,

- Check python version: `python3 --version`
- If not installed: `sudo apt-get install python3.6`
- Install pygame: `sudo apt-get install python3-pygame`
- Go to the root directory and run `python3 main.py`

If user wants to play against Alpha-Beta pruning, user has to change instance variable self.player to 2, self.second
to False, self.AIReadyToMove to False modify the parameter of function calling in start() in main.py():
set self.game.AIMove(0), and call
if __name__ == '__main__':
	engine = Game_Engine()
	engine.start()
at the end of main.py().

If user wants to play against NegaScout, user has to change instance variable self.player to 2, self.second
to False, self.AIReadyToMove to False modify the parameter of function calling in start() in main.py():
set self.game.AIMove(1), and call
if __name__ == '__main__':
	engine = Game_Engine()
	engine.start()
at the end of main.py().

If user wants to see Alpha-Beta pruning plays against NegaScout:
- If NegaScout plays first, then set instance variable self.player = 1, self.AIReadyToMove = False, self.second = True,
  self.secondAIReadyToMove = True in Othello.py and call
  if __name__ == '__main__':
	engine = Game_Engine()
	engine.AIstart()
at the end of main.py(). Alpha-Beta is always player2, and NegaScout is always player1 no matter the order they playing.

- If Alpha-Beta pruning plays first, then set instance variable self.player = 2, self.AIReadyToMove = True, self.second = True,
  self.secondAIReadyToMove = False in Othello.py and call
  if __name__ == '__main__':
	engine = Game_Engine()
	engine.AIstart()
at the end of main.py(). Alpha-Beta is always player2, and NegaScout is always player1 no matter the order they playing.

- To get sorted nodes for sucessors instead of random, change everywhere calling findSuccessorBoards(Board, player)
  to getSortedNodes(Board, player) in the NegaScout and MiniMax function.

