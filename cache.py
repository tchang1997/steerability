import atexit
import os
import pathlib
import pickle
import signal

def catch_kbint(*args):
    print("Ctrl-C detected - disabled during cache saving to prevent corruption. If you absolutely need to, please use Ctrl-Z + `kill` to terminate the program.")

class GoalspaceCache(object):

    def __init__(self, goalspace_cache):
        self.cache = self._get_or_create_cache(goalspace_cache)
        atexit.register(self.cleanup)

    def _get_or_create_cache(self, goalspace_cache):
        self.goalspace_path = pathlib.Path.cwd() / "cache" / "goalspace" / (goalspace_cache + ".pkl")
        if os.path.isfile(self.goalspace_path):
            with open(self.goalspace_path, "rb") as f:
                try:
                    self.cache = pickle.load(f)
                except Exception as e:
                    print("Cache is corrupted! Path:", self.goalspace_path)
                    self.cache = {}
        else:
            self.cache = {}
        return self.cache

    def _save_cache(self):
        print("Saving cache to", self.goalspace_path)
        with open(self.goalspace_path, "wb") as f:
            pickle.dump(self.goalspace_cache, f)

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def cleanup(self):
        print("Saving cache to", self.goalspace_path)
        orig_handler = signal.signal(signal.SIGINT, catch_kbint)
        try:
            with open(self.goalspace_path, "wb") as f:
                pickle.dump(self.cache, f)
        finally:
            signal.signal(signal.SIGINT, orig_handler)
            

