# Monte Carlo Tree Search (MCTS)

A comprehensive implementation of the Monte Carlo Tree Search algorithm in Python, featuring a modular design that supports various games and decision-making problems. This implementation emphasizes clarity, performance, and extensibility.

## Features

- **Pure Python Implementation**: No external dependencies required
- **Game-Agnostic Design**: Easy to adapt to any turn-based game
- **Efficient Search**: Optimized node selection and expansion
- **Parallelization**: Support for parallel simulations
- **Visualization**: Built-in tools for tree visualization
- **Multiple Game Examples**: Ready-to-run implementations of popular games

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Overview](#algorithm-overview)
- [Examples](#examples)
- [Custom Game Integration](#custom-game-integration)
- [Advanced Usage](#advanced-usage)
- [Performance Tuning](#performance-tuning)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- NumPy (for some examples)
- Matplotlib (for visualization)
- tqdm (for progress bars)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/monte-carlo-tree-search.git
cd monte-carlo-tree-search

# Install with pip
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from mcts import MCTS
from games.tictactoe import TicTacToeGame

# Initialize game and MCTS
game = TicTacToeGame()
mcts = MCTS(game)

# Run search
best_action = mcts.search(iterations=1000)
print(f"Best action: {best_action}")

# Get action probabilities
action_probs = mcts.get_action_probabilities()
print(f"Action probabilities: {action_probs}")
```

### Running Examples

```bash
# Tic-tac-toe against AI
python examples/play_tic_tac_toe.py

# Connect Four with visualization
python examples/visualize_connect_four.py

# Custom game implementation
python examples/custom_game.py
```

## Algorithm Overview

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm that combines the precision of tree search with the generality of random sampling. It is particularly effective in games with large branching factors.

### The MCTS Process

1. **Selection**: Traverse the tree from root to leaf node using UCB1 selection
2. **Expansion**: Add child nodes if leaf is not terminal
3. **Simulation**: Play out random moves until terminal state
4. **Backpropagation**: Update statistics along the traversed path

### Key Components

- **Node**: Represents a game state and stores statistics
- **Tree**: Manages the game tree and node relationships
- **Selection Policy**: Determines how to traverse the tree (UCB1 by default)
- **Simulation Policy**: Defines how to simulate random playouts

## Examples

### Tic-tac-toe

```python
from mcts import MCTS
from games.tictactoe import TicTacToeGame

# Initialize game and MCTS
game = TicTacToeGame()
mcts = MCTS(
    game,
    iterations=1000,       # Number of MCTS iterations
    exploration_weight=1.4,  # Balance exploration/exploitation
    time_limit=None,       # Optional time limit in seconds
    parallel=False         # Enable parallel simulations
)

# Get best action
best_action = mcts.search()
print(f"Best action: {best_action}")

# Visualize the search tree
mcts.visualize_tree()
```

### Connect Four

```python
from mcts import MCTS
from games.connect_four import ConnectFourGame

# Initialize game with custom board size
game = ConnectFourGame(rows=6, columns=7, win_length=4)
mcts = MCTS(game, iterations=2000)

# Interactive play
while not game.is_game_over():
    if game.current_player == 1:  # AI's turn
        action = mcts.search()
    else:  # Human's turn
        game.print_board()
        action = int(input("Enter column (0-6): "))
    
    game.take_action(action)
    mcts.update_root(action)

game.print_board()
print(f"Game over! Result: {game.get_result()}")
```

## Custom Game Integration

To implement your own game, create a class that implements the following interface:

```python
class Game:
    def get_legal_actions(self):
        """Return a list of legal actions from the current state."""
        pass
        
    def take_action(self, action):
        """Apply an action and return a new game state."""
        pass
        
    def is_game_over(self):
        """Return True if the game is over."""
        pass
        
    def get_result(self):
        """
        Return the game result from the current player's perspective.
        Returns:
            float: 1.0 for win, 0.5 for draw, 0.0 for loss
        """
        pass
        
    def get_current_player(self):
        """Return the current player (0 or 1)."""
        pass
        
    def copy(self):
        """Return a deep copy of the game state."""
        pass
        
    def __str__(self):
        """Return a string representation of the game state."""
        pass
```

## Advanced Usage

### Parallelization

For CPU-bound workloads, enable parallel simulations:

```python
from mcts import MCTS
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    mcts = MCTS(game, iterations=10000, parallel=True, executor=executor)
    best_action = mcts.search()
```

### Custom Selection Policy

Implement a custom selection policy by subclassing `SelectionPolicy`:

```python
from mcts import SelectionPolicy
import math

class UCT(SelectionPolicy):
    def __init__(self, exploration_weight=math.sqrt(2)):
        self.exploration_weight = exploration_weight
    
    def select_child(self, node):
        log_N = math.log(node.visits) if node.visits > 0 else 0
        
        def ucb(n):
            if n.visits == 0:
                return float('inf')
            return (n.wins / n.visits) + self.exploration_weight * math.sqrt(
                log_N / n.visits
            )
            
        return max(node.children, key=ucb)
```

### Saving and Loading Trees

```python
import pickle

# Save tree
with open('mcts_tree.pkl', 'wb') as f:
    pickle.dump(mcts.tree, f)

# Load tree
with open('mcts_tree.pkl', 'rb') as f:
    mcts.tree = pickle.load(f)
```

## Performance Tuning

### Key Parameters

1. **Iterations**: More iterations → better moves, but slower
2. **Exploration Weight**: Higher values favor exploration
3. **Time Limit**: Alternative to fixed iterations
4. **Parallelization**: Speed up with multiple cores
5. **Simulation Depth**: Limit simulation steps for complex games

### Optimization Tips

- **Reuse Trees**: Update the root instead of creating new trees
- **Early Termination**: Stop early if a clear best move is found
- **Progressive Widening**: Limit child nodes in early stages
- **Domain Knowledge**: Guide simulations with heuristics

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [A Survey of Monte Carlo Tree Search Methods](https://www.cambridge.org/core/journals/ieee-transactions-on-computational-intelligence-and-ai-in-games/article/survey-of-monte-carlo-tree-search-methods/1D0B3B1E6E2A8F5E8E5F2F3F3F3F3F3)
- [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://hal.inria.fr/inria-00116992/)
- [The Monte Carlo Revolution in Go](https://www.americanscientist.org/article/the-monte-carlo-revolution-in-go)

## References

1. Kocsis, L., & Szepesvári, C. (2006). Bandit based Monte-Carlo Planning. In European Conference on Machine Learning.
2. Browne, C. B., et al. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE Transactions on Computational Intelligence and AI in Games.
3. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature.
