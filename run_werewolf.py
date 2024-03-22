"""
    Automatic Running Script of Werewolf Game
    Developed by Yuzhuang Xu, Tsinghua University
    v1.0 2023-05-04
    Base Platform: ChatArena(https://github.com/chatarena/chatarena)
"""

import argparse
import os
import json
from chatarena.arena import Arena, TooManyInvalidActions
from chatarena.config import ArenaConfig
from chatarena.message import Message
from chatarena.backends.human import HumanBackendError
from chatarena.environments import Environment, Werewolf


def main():
    parser = argparse.ArgumentParser(description="The command-line parameter of werewolf game.")
    
    parser.add_argument("--current-game-number", type=int, default=0, help="this is the serial number of current game, must a integer")
    parser.add_argument("--message-window", type=int, default=10, help="number of the newest message for driving game reasoning")
    parser.add_argument("--answer-topk", type=int, default=5, help="number of the retrieval answers for choosing")
    parser.add_argument("--exps-retrieval-threshold", type=float, default=0.6, help="experiences whose reflexion similarity larger than it will be recalled")
    parser.add_argument("--similar-exps-threshold", type=float, default=0.1, help="experiences whose similarity difference is less than it will be omited")
    parser.add_argument("--max-tokens", type=int, default=100, help="maximum tokens of each generation")
    parser.add_argument("--retri-question-number", type=int, default=5, help="number of questions from question history")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature hyper-parameter of generation model")
    parser.add_argument("--use-api-server", type=int, default=0, help="use the self-developed api server for anytime calling")

    parser.add_argument("--save-exps-incremental", action="store_true", default=False, help="save all experiences defore this piece of game in a file")
    parser.add_argument("--use-crossgame-exps", action="store_true", default=False, help="use the cross-trajectory experiences of different games")
    parser.add_argument("--use-crossgame-ques", action="store_true", default=False, help="use the cross-trajectory questions of different games")
    parser.add_argument("--human-in-combat", action="store_true", default=False, help="enable Player 1 with human")
    
    parser.add_argument("--environment-config", type=str, default="./examples/werewolf.json", help="json file that define the rule and players")
    parser.add_argument("--role-config", type=str, default="./config/1.json", help="json file that define the number of roles")
    parser.add_argument("--exps-path-to", type=str, help="path of saving binary files of experiences")
    parser.add_argument("--ques-path-to", type=str, help="path of saving binary files of questions")
    parser.add_argument("--logs-path-to", type=str, default="./logs", help="path of saving log files of competitive talking")
    parser.add_argument("--load-exps-from", type=str, help="path of experience files for loading")
    parser.add_argument("--load-ques-from", type=str, help="path of question files for loading")
    parser.add_argument("--who-use-exps", nargs='+', help="a list of roles that will use experiences")
    parser.add_argument("--who-use-ques", nargs='+', help="a list of roles that will use question histories")
    
    args = parser.parse_args()
    
    if args.exps_path_to:
        os.makedirs(args.exps_path_to, exist_ok=True)
    if args.ques_path_to:
        os.makedirs(args.ques_path_to, exist_ok=True)
    os.makedirs(args.logs_path_to, exist_ok=True)
    with open(os.path.join(args.logs_path_to, str(args.current_game_number) + ".md"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg} : {getattr(args, arg)}  " + "\n")
        f.write("\n")

    with open(args.environment_config, "r") as f:
        config = json.load(f)
    moderator_config = {
        "role_desc": "",
        "global_prompt": config["global_prompt"],
        "terminal_condition": "",
        "backend": {
            "backend_type": "openai-chat",
            "temperature": 0.2,
            "max_tokens": 100
        }
    }
    env_config = {
            "env_type": "werewolf",
            "parallel": False,
            "moderator": moderator_config,
            "moderator_visibility": "all",
            "moderator_period": "turn"
        }
    player_configs = []
    for i in range(len(config["players"])):
        player_name = f"Player {i + 1}"
        role_desc, backend_type, temperature, max_tokens = config["players"][i]["role_desc"], \
            config["players"][i]["backend"]["backend_type"], config["players"][i]["backend"]["temperature"], \
            config["players"][i]["backend"]["max_tokens"]
        player_config = {
            "name": player_name,
            "role_desc": role_desc,
            "global_prompt": config["global_prompt"],
            "backend": {
                "backend_type": backend_type,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        player_configs.append(player_config)
        
    arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config), args)
    
    while True:
        try:
            timestep = arena.step(args)
        except TooManyInvalidActions as e:
            timestep = arena.current_timestep
            timestep.observation.append(
                Message("System", "Too many invalid actions. Game over.", turn=-1, visible_to="all"))
            timestep.terminal = True  
            
        if timestep.terminal == True:
            break
        
    if args.exps_path_to:
        arena.environment.message_pool.save_exps_to(args.save_exps_incremental)
    if args.ques_path_to:
        arena.environment.question_pool.save_ques_to(args.save_exps_incremental)


if __name__ == "__main__":
    main()