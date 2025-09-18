import sys
import math
import random
######################################
# Todo
# 
# 1. Monte Carlo Tree Search
##################################### 


WIN_LINES = [
    [(0,0),(0,1),(0,2)],  # rows
    [(1,0),(1,1),(1,2)],
    [(2,0),(2,1),(2,2)],
    [(0,0),(1,0),(2,0)],  # cols
    [(0,1),(1,1),(2,1)],
    [(0,2),(1,2),(2,2)],
    [(0,0),(1,1),(2,2)],  # diagonals
    [(0,2),(1,1),(2,0)]
]


class GameBoard:

    def __init__(self):

        self.entries = [[0, 0, 0], [0, 0, 0], [0, 0 ,0]]
        self.state = 0

        self.minmax_nodes = 0
        self.ab_nodes = 0
        self.ab_prunes = 0
        # State 0: Game playing
        # State 1: Player 1 wins
        # State 2: Player 2 wins
        # State 3: draw

    def print_bd(self):

        for i in range(3):
            for j in range(3):
                print(self.entries[i][j],end='')
            print('')



    def checkwin(self) -> int:
        
        for line in WIN_LINES:
            vals = [self.entries[r][c] for r,c in line]
            if vals == [1, 1, 1]:   
                return 1
            if vals == [2, 2, 2]:
                return 2
            
        if any(0 in row for row in self.entries):
            return 0

        return 3
    
    def check_nextplayer(self, bd = None):
        #count how many 1 and 2 in bd
        count_1 = sum(cc == 1 for row in bd for cc in row)
        count_2 = sum(cc == 2 for row in bd for cc in row)
        
        if count_1 > count_2:
            return 2
        else:
            return 1
        
    def getmoves(self):
        return [(r,c) for r in range(3) for c in range(3) if self.entries[r][c]==0] # all possible position where the board is empty
    
    def copy(self):
        new_board = GameBoard()
        new_board.entries = [row[:] for row in self.entries]
        return new_board
    

    def minmax(self, bd=None, depth=0):
        # set default player to 1(X) cause it is the first player
        # score for x: +, score for o:-
        # return (move, score)
        self.minmax_nodes += 1 
        
        result = self.checkwin()
        if result == 1: 
            return None, 10-depth # x win, prefer faster wins 
        if result == 2: 
            return None, depth-10 # o win, prefer slower losses 
        if result == 3: 
            return None, 0 #draw
        


        moves = [(r,c) for r in range(3) for c in range(3) if bd[r][c]==0] # all possible position where the board is empty
        player = self.check_nextplayer(bd)
        

        if player == 1: # x's turn, maximize
            best = -1e9 # initilize to very small number
            move = None 
            for r,c in moves:
                bd[r][c]=1 # if x plays here
                _,score=self.minmax(bd,depth+1) # o's turn 
                bd[r][c]=0 # undo move
                
                if score>best: # pick the move with the largest score, update the best move!!
                    best,move=score,(r,c)
            return move,best
        
        else: # o's turn, minimize     
            best = 1e9 # initilize to very large number
            move=None
            for r,c in moves:
                bd[r][c]=2
                _,score=self.minmax(bd,depth+1) # x's turn
                bd[r][c]=0
                
                if score<best: # pick the move with the smallest score, update the best move!!
                    best,move=score,(r,c)
            return move,best
        

        
    def alphabeta(self, bd=None, depth=0,alpha = -math.inf, beta = math.inf):
      
        self.ab_nodes += 1
        result = self.checkwin()
        if result == 1: 
            return None, 10-depth # x win, prefer faster wins 
        if result == 2: 
            return None, depth-10 # o win, prefer slower losses 
        if result == 3: 
            return None, 0 #draw
        


        moves = [(r,c) for r in range(3) for c in range(3) if bd[r][c]==0] # all possible position where the board is empty
        player = self.check_nextplayer(bd)
        

        if player == 1: # x's turn, maximize
            best = -1e9 # initilize to very small number
            move = None 
            for r,c in moves:
                bd[r][c]=1 # if x plays here
                _,score=self.alphabeta(bd,depth+1,alpha, beta) # o's turn 
                bd[r][c]=0 # undo move
                
                if score>best: # pick the move with the largest score
                    best,move=score,(r,c)
                
                ## alpha pruning
                alpha = max(alpha, best)
                if alpha >= beta:  # beta cut
                    self.ab_prunes += 1
                    break
            return move,best
        
        else: # o's turn, minimize     
            best = 1e9 # initilize to very large number
            move=None
            for r,c in moves:
                bd[r][c]=2
                _,score=self.alphabeta(bd,depth+1,alpha, beta) # x's turn
                bd[r][c]=0
                
                if score<best: # pick the move with the smallest score
                    best,move=score,(r,c)

                beta = min(beta, best)
                if alpha >= beta:  # alpha cut
                    self.ab_prunes += 1
                    break
            return move,best


class MCTSNode:
    def __init__(self, bd: GameBoard, parent: None, action: None):
        self.bd = bd             
        self.parent = parent
        self.action = action # action that led to this node          
        self.children = [] # list of child nodes
        self.possible_moves = bd.getmoves()  # moves that can be played from this node
        self.visits = 0
        self.wins = 0.0         

    def is_fully_expanded(self):
        return len(self.possible_moves) == 0
    
    def is_terminal(self):
        return self.bd.checkwin() != 0
    
def apply_action(bd:GameBoard, action, player):
        r,c = action        
        new_bd = bd.copy()
        new_bd.entries[r][c] = player
        return new_bd


class MCTS:
    def __init__(self, c = math.sqrt(2)):
        self.c = c

    
    def uct_select(self, node:MCTSNode, c = None) -> MCTSNode:
        # node: current node
        # return: child node with highest UCT value
        if c is None:
            c = self.c
        return max(node.children, key=lambda child: (child.wins / child.visits) + c*math.sqrt(math.log(node.visits)/child.visits))

    def expand(self, node:MCTSNode) -> MCTSNode:
        # node: current node
        # return: new child node after applying one of the possible moves

        action = node.possible_moves.pop()
        r,c = action

        child_bd = node.bd.copy()
        player = child_bd.check_nextplayer(child_bd.entries)
        child_bd = apply_action(child_bd, action, player)

        child_node = MCTSNode(child_bd, parent=node, action = action)
        node.children.append(child_node)
        return child_node
    
    def rollout(self,bd:GameBoard) -> int:
        # bd: current board
        # return: score on the same (+1: for winner player)
        
        rollout_bd = bd.copy()

        while rollout_bd.checkwin() == 0:
            next_player = rollout_bd.check_nextplayer(rollout_bd.entries)
            actions = rollout_bd.getmoves()
            if not actions:
                break
            action =random.choice(actions)
            rollout_bd = apply_action(rollout_bd, action, next_player)
        
        winner = rollout_bd.checkwin()
        if winner ==1 or winner ==2:
            return +1
        else:
            return 0
        
    def backpropagate(self, node:MCTSNode, reward:int):
        current = node
        while current is not None:
            current.visits += 1
            current.wins += reward
            # switch perspective
            reward = -reward

            #propagate to parent
            current = current.parent
    
    def search(self,root:MCTSNode, iter = 2000):
        # based on the rood board, run MCTS and return the best action
        #root = MCTSNode(root_bd, parent=None, action=None)
        root_bd = root.bd
        if root_bd.checkwin() != 0:
            raise ValueError("Game is over")
        
        for _ in range(iter):
            node = root

            # selection
            while (not node.is_terminal()) and node.is_fully_expanded():
                node = self.uct_select(node, c= self.c)
            
            # expansion
            if (not node.is_terminal()) and (not node.is_fully_expanded()):
                node = self.expand(node)

            # simulation 
            reward = self.rollout(node.bd)

            # backpropagation
            self.backpropagate(node, reward)
        self.c = 0
        best_child = self.uct_select(root,c=0)
        return best_child.action

def MCTS_move(root_state: GameBoard, iterations=2000):
    mcts = MCTS()
    root_node = MCTSNode(bd = root_state, parent= None, action=None)
    
    best_action = mcts.search(root_node, iter=iterations)
    player = root_state.check_nextplayer(root_state.entries)
    next_state = apply_action(root_state, best_action, player)
    
    return best_action, player, next_state
            

class TicTacToeGame:

    def __init__(self):

        self.gameboard = GameBoard()
        self.turn = 1 # first player is 1
        self.turnnumber = 1

    def x_or_o():
        valid_input = False

        while valid_input == False:
            user_input = input("Choose X or O:").upper()
            if user_input == "X" or user_input == "O":
                print("Valid Input")
                valid_input = True
            else:
                print("Please enter valid input")
                continue
        
        return user_input

    def playturn(self):
        if (player_choice == "X" and self.turnnumber <= 1):
            print("You chose X!")
            self.turn = 1
        elif player_choice == "O" and self.turnnumber <= 1:
            print("You chose O!")
            self.turn = 2

        print("Turn number: ", self.turnnumber)
        self.turnnumber += 1
        self.alpha = -math.inf
        self.beta = math.inf
        
        self.gameboard.print_bd()

        if self.turn == 1:
            print("Human, please choose a space!")
            validinput = False
            
            while validinput == False:
                user_input = input("Enter two numbers separated by a comma: ")
                if user_input == "q":
                    break
                try:
                    humanrow, humancol = map(int, map(str.strip, user_input.split(',')))
                    if humanrow < 3 and humancol < 3:
                        if self.gameboard.entries[humanrow][humancol] == 0:
                            validinput = True
                        else:
                            raise ValueError
                    else:
                        raise ValueError
                except:
                    print("Please use a valid input") 
                    continue
                 
            self.gameboard.entries[humanrow][humancol] = 1
            self.turn = 2
        else:
            print("AI is thinking...")
            best_action, next_player, next_bd = MCTS_move(self.gameboard, iterations=1000)            
            self.gameboard.entries[best_action[0]][best_action[1]] = 2
            self.turn = 1
            

            # move, score = self.gameboard.minmax(self.gameboard.entries)
            # #move, score = self.gameboard.alphabeta(self.gameboard.entries,self.alpha,self.beta)
            # print("AI chooses move: ", move, " with score: ", score)
            # self.gameboard.entries[move[0]][move[1]] = 2
            # self.turn = 1



game = TicTacToeGame()

player_choice = TicTacToeGame.x_or_o()

while game.gameboard.state == 0:
    game.playturn()
    game.gameboard.state = game.gameboard.checkwin()
    print(' ')

game.gameboard.print_bd()
if game.gameboard.state == 1:
    print("Player 1 wins!")
elif game.gameboard.state == 2:
    print("Player 2 wins!")
else:
    print("The game is a draw!")

###########################################
# For testing minmax and alphabeta pruning
# Choose different board states and see how many nodes are prunned
# #############################################
# gb = GameBoard()
# gb.entries = [   [0, 0, 0],
#                 [0, 0, 0],
#                 [0, 0, 0]]
# gb.print_bd()
# mm_move, mm_score = gb.minmax(gb.entries,0)
# ab_move, ab_score = gb.alphabeta(gb.entries,0,-math.inf,math.inf)
# print("Minmax: move=(%d,%d), score=%d, nodes=%d" %(mm_move[0],mm_move[1], mm_score, gb.minmax_nodes))
# print("AlphaBeta: move=(%d,%d), score=%d, nodes=%d, prunes = %d" %(ab_move[0],ab_move[1], ab_score, gb.ab_nodes, gb.ab_prunes))
