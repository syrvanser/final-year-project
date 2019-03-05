import logging
from games import *
from agents import *
from nnet_wrapper import NNetWrapper
logger = logging.getLogger(__name__)

#TODO fix drop, +k
def play():
    rounds = 1
    white_wins = 0
    for i in range (1, rounds+1):
        begin = time.time()
        print('Game {0}/{1}'.format(i, rounds))
        g = Game()
        agent1 = RandomAgent()
        agent2 = BasicMCTSAgent(limit = 20, max_depth = 30)
        while True:
            current_agent = agent1 if g.game_state.colour == 'W' else agent2
            current_agent.act(g)
            #print(g.game_state.print())
            logger.info(g.game_state.print(flip=g.game_state.colour== 'B'))
            if(g.game_state.game_ended()):
                if g.game_state.colour == 'B':
                    white_wins +=1
                print('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(agent1.__class__.__name__, white_wins, white_wins/i*100, agent2.__class__.__name__, i-white_wins, (i-white_wins)/i*100, time.time()-begin))
                logger.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(agent1.__class__.__name__, white_wins, white_wins/i*100, agent2.__class__.__name__, i-white_wins, (i-white_wins)/i*100, time.time()-begin))
                break
        

def play_one_game():
    train_examples = []
    g = Game()
    agent = NNetMCTSAgent()
    step = 0
    while True:
        step +=1
        
        



def train_neural_net():
    nnet = NNetWrapper()
    play_one_game()

def saveTrainExamples(self, train_examples, folder = 'checkpoints', filename='examples.data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, filename)
    with open(filename, 'wb+') as f:
        Pickler(f).dump(train_examples)

def loadTrainExamples(self, folder='checkpoints', filename='examples.data'):
    examplesFile = os.path.join(folder, filename)
    with open(examplesFile, 'rb') as f:
        return Unpickler(f).load()

if __name__ == '__main__':
    #print(Game.get_coordinates(0,3,0,2))
    random.seed(1)
    logging.basicConfig(format='%(asctime)s %(name)-14s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='game.log',
                        filemode='w',
                        level=logging.INFO)
    np.set_printoptions(threshold=np.nan)

    play()
  
