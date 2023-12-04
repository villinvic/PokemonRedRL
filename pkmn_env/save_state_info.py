from pathlib import Path


class PokemonStateInfo:

    def __init__(
            self,
            *,
            save_path: Path = None,
            latest_opp_level: int = None,
            visited_maps: set = None,

    ):
        self.save_path = save_path
        self.latest_opp_level = latest_opp_level
        self.visited_maps = visited_maps

    def to_dict(self):

        return {
            "save_path": self.save_path,
            "latest_opp_level": self.latest_opp_level,
            "visited_maps": self.visited_maps
        }

    def get_save_path(self):
        return self.save_path.parent / (self.save_path.name + '.state')

    def get_info_path(self):
        return self.save_path.parent / (self.save_path.name + '_info.pkl')

    def from_dict(self, d):

        for k, v in d.items():
            setattr(self, k, v)

    @staticmethod
    def dict_to_env(env, d):

        for k, v in d.items():
            setattr(env, k, v)

    def send_to_env(self, env):
        self.dict_to_env(env, self.to_dict())
