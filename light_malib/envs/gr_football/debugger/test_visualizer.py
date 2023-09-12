# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from light_malib.envs.gr_football.tools.tracer import MatchTracer
from light_malib.envs.gr_football.game_graph.game_graph import GameGraph
from light_malib.envs.gr_football.debugger.visualizer import Visualizer
import numpy as np

# tracer = MatchTracer.load("temp/random_play.trace")
tracer=MatchTracer.load_from_official_trace("/tmp/football_render/competition_232948/episode_done_20230908-151201056126.dump")
game_graph = GameGraph(tracer)
print(game_graph)
visualizer = Visualizer(tracer, disable_RGB=False, disable_reward=False)
visualizer.run()
