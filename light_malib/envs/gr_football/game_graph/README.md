# Game Graph Data Structure

You can generate a game graph by using `MatchTracer` instance to record the match then passing in the `GameGraph` class.

```python
from tracer import MatchTracer
from game_graph import GameGraph

tracer=MatchTracer()

# use tracer to record the match.
# ...

game_graph=GameGraph(tracer)

# access attributes of the game graph
# ...
```

You can get access to the following attributes in the Game Graph.

Indexing
* nodes: node list
* chains: chain list
* subgames: subgame list

Inverse Indexing
* step2node: mapping from step to node
* step2chain: mapping from step to chain
* step2subgame: mapping from step to subgame

Events:
* goals: mapping from event beginning step to goal event
* passings: mapping from event beginning step to passing event
* losing_balls: mapping from event beginning step to losing ball event
* intercepting_balls: mapping from event beginning step to intercepting ball event

[Return to the main README](../../../../README.md#google-football-game-graph)