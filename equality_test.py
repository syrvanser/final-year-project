from games import Game, GameState

g1 = Game()
print(g1.game_state.__hash__())
g2 = Game()
print(g2.game_state.__hash__())
print(g1.game_state == g2.game_state)
print(g1.game_state.__eq__(g2.game_state))
h1 = print(hash(str(GameState.board_to_plane_stack(g1.game_state.board, g1.game_state.hand1, g1.game_state.hand2, g1.game_state.repetitions, g1.game_state.colour, g1.game_state.move_count))))
h2 = print(hash(str(GameState.board_to_plane_stack(g2.game_state.board, g2.game_state.hand1, g2.game_state.hand2, g2.game_state.repetitions, g2.game_state.colour, g2.game_state.move_count))))
print (h1 == h2)
print (hash(str(GameState.board_to_plane_stack(g1.game_state.board, g1.game_state.hand1, g1.game_state.hand2, g1.game_state.repetitions, g1.game_state.colour, g1.game_state.move_count))) == hash(str(GameState.board_to_plane_stack(g2.game_state.board, g2.game_state.hand1, g2.game_state.hand2, g2.game_state.repetitions, g2.game_state.colour, g2.game_state.move_count))))

