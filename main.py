import logging
from games import *
from agents import *
logger = logging.getLogger(__name__)

#TODO fix drop, +k
def main():
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='game.log',
                        filemode='w',
                        level=logging.INFO)

    np.set_printoptions(threshold=np.nan)
    rounds = 1
    white_wins = 0
    for i in range (1, rounds+1):
        begin = time.time()
        print('Game {0}/{1}'.format(i, rounds))
        g = Game()
        agent1 = RandomAgent()
        agent2 = MCTSAgent(limit = 50, max_depth = 40)
        while True:
            current_agent = agent1 if g.game_state.colour == 'W' else agent2
            current_agent.act(g)
            logger.info(g.game_state.print())
            if(g.game_state.game_ended()):
                logger.info('Game end: ' + current_agent.__class__.__name__ + ' wins')
                if g.game_state.colour == 'B':
                    white_wins +=1
                print('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(agent1.__class__.__name__, white_wins, white_wins/i*100, agent2.__class__.__name__, i-white_wins, (i-white_wins)/i*100, time.time()-begin))
                break
        

if __name__ == '__main__':
    #print(Game.get_coordinates(0,3,0,2))
    random.seed(1)
    main()
  
