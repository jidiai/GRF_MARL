# Single-Step Visual Debugger

1. The 1st Method: use the official `.dump` file by set `write_full_episode_dumps=True` in the environment. 
    use `load_from_official_trace()` function to read the `.dump` file and pass into visualizer for replay.
    ```python
    from tracer import MatchTracer
    from v.visualizer import Visualizer

    tracer=MatchTracer.load_from_official_trace("data/episode_done_20220426-213219251284.dump")
    # disable_RGB=True means don't load RGB frames, which might be slower.
    visualizer=Visualizer(tracer,disable_RGB=True)
    visualizer.run()
    ```
2. The 2nd Method: use `MatchTracer` instance to record the match (you may refer to `test_tracer.py` for another example).
    ```python
    from tracer import MatchTracer
    class FooballEnv:
        def __init__(self):
            self._env=...

        def reset(self):
            # no_frame=True means not to store RGB frames.
            self.tracer=MatchTracer(no_frame=True)
            self._observations=self._env.reset()
        
        def step(self,actions):
            self.tracer.update(self._observations,actions)
            self._observations,reward,done,info=self._env.step(actions)
            if done:
                self.tracer.update(self._observations)
                self.tracer.save(fn=...)
    ```

    use `load()` function to read `.tracer` file and pass into visualizer for replay.
    ```python
    from tracer import MatchTracer
    from v.visualizer import Visualizer

    tracer=MatchTracer.load("data/random_play_trace.pkl")
    # disable_RGB=True means don't load RGB frames, which might be slower.
    visualizer=Visualizer(tracer,disable_RGB=True)
    visualizer.run()
    ```

NOTE: If you want to watch RGB frames as well in replay, you need to set `render=True` in the environment.

[Return to the main README](../../../../README.md#google-reseach-football-toolkit)