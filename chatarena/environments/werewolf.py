from typing import List, Dict, Union, Tuple
import random
import re
import json
import logging
import sys

from .base import Environment, TimeStep
from ..message import Message, MessagePool, Question, QuestionPool
from ..agent import SIGNAL_END_OF_CONVERSATION

logging.basicConfig(level=logging.DEBUG)


class Werewolf(Environment):
    type_name = "werewolf"

    def __init__(self, args, player_names: List[str], topic_codes: Dict[str, List[str]] = None, **kwargs):
        super().__init__(player_names=player_names, topic_codes=topic_codes, **kwargs)
        
        self.args = args
        self.message_pool = MessagePool(args)
        self.question_pool = QuestionPool(args)

        if not args or (args and not args.role_config):
            with open("./config/1.json", "r") as f:
                self._character_config = {k: v for k, v in json.load(f).items() if v != 0}
        else:
            with open(args.role_config, "r") as f:
                self._character_config = {k: v for k, v in json.load(f).items() if v != 0}
        self._alive_list = self.player_names + ["pass"]
        self._characters = [k for k, v in self._character_config.items() for _ in range(v)]
        random.shuffle(self._characters)        # ["werewolf", "seer", "witch", "villager"...]
        self._is_alive = [True for _ in range(len(self._characters))]
        self._identity_mapping = {              # {"werewolf":[2,3], "villager":[1]...}
            "werewolf": [],
            "villager": [],
            "guard": [],
            "witch": [],
            "seer": [],
            "hunter": [],
            "idiot": [],
            "thief": [],
            "cupid": [],
            "girl": [],
            "sheriff": [],
            "elder": [],
            "scapegoat": [],
            "piper": []
        }
        for i, x in enumerate(self._characters):
            self._identity_mapping[x].append(i)
        self.werewolf_numbers = len(self._identity_mapping["werewolf"])
        # The following: [2,3,1...]
        self._night_order = [x for k, v in self._identity_mapping.items() if k != "villager" and len(v) > 0 for x in v]
        # The following: [3,4,1...] must be shuffled exery day!
        self._day_order = [i for i in range(len(self._characters))]
        random.shuffle(self._day_order)

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "night"           # night, daytime
        self._number_of_nights = 0
        self._initialized = False
        self._players_votes = None
        self._lives = 0
        self._number_of_rounds = 0
        self._guard_to = None
        self._witch_antidote = True
        self._witch_poison = True
        self._witch_antidote_to = None
        self._witch_poison_to = None
        self._current_first_alive = -1
        self._killed_list = []
        self._night_kill_list = []

        # self.reset()  # To initialize the game

    def _print_infos(self):
        print(f"alive list: {self._alive_list}", file=sys.stderr)
        print(f"is alive: {self._is_alive}", file=sys.stderr)
        print(f"identity mapping: {str(self._identity_mapping)}", file=sys.stderr)
        print(f"night order: {self._night_order}", file=sys.stderr)
        print(f"day order: {self._day_order}", file=sys.stderr)

    def get_next_player(self) -> str:
        """
            get the next player
        """
        if self._number_of_rounds == 9999:             # last statement
            return self.player_names[self._next_player_idx]

        if self._current_phase == "daytime":
            return self.player_names[self._day_order[self._next_player_idx]]
        elif self._current_phase == "night":
            return self.player_names[self._night_order[self._next_player_idx]]

    def reset(self):
        self.message_pool.reset()

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "night"
        self._number_of_nights = 0
        self._lives = len(self._characters)
        self._players_votes = {name: 0 for name in self.player_names}
        self._players_votes["pass"] = 0

        self._description = [str(len(v)) + ' ' + k + "(s), " for k, v in self._identity_mapping.items() if len(v) > 0]
        _list = list(self._description[-1])
        _list.pop()
        _list[-1] = '.'
        self._description[-1] = ''.join(_list)
        self._description = ''.join(self._description)

        self._print_infos()
        self._moderator_speak(f"Now the game starts! In this game, we have {self._description}")
        for i, player in enumerate(self.player_names):
            self._moderator_speak(f"You are {self._characters[i]}!", visible_to=player, importance=1)
        self._moderator_speak("It's dark, everyone close your eyes. I will talk with you/your team secretly at night.")
        werewolves = ', '.join([self.player_names[i] for i in self._identity_mapping["werewolf"]])
        print(f"wolves: {werewolves}", file=sys.stderr)
        print(f"alive list: {self._alive_list}", file=sys.stderr)
        self._moderator_speak(f"Werewolves, please open your eyes! "
                              f"I secrecly tell you that {werewolves} are all of the {len(self._identity_mapping['werewolf'])} werewolves! "
                              f"Keep in mind you are teammates. The rest players are not werewolves. Now vote and tell your teammates which of the players should be killed tonight. The first werewolf, "
                              f"you, randomly choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                              # f"For example: I choose Player...",
                              visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                              importance=5)
        self._current_turn = 1

        self._initialized = True
        init_timestep = TimeStep(observation=self.get_observation(),
                                 reward=self.get_zero_rewards(),
                                 terminal=False)

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """
            get observation for the player
        """
        n_last = self.args.message_window if self.args and self.args.message_window else 10
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            # return self.message_pool.get_visible_messages(player_name, turn=self._current_turn)
            return self.message_pool.get_last_k_messages(player_name, self._current_turn, n_last)

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all", importance = 1):
        """
            moderator say something
        """
        message = Message(agent_name="Moderator", content=text, turn=self._current_turn, visible_to=visible_to, importance=importance)
        self.message_pool.append_message(message)

    def _get_number_of_people(self) -> Tuple[int, int, int]:
        _all = len(self._characters)
        _live = self._lives
        _werewolf = len(self._identity_mapping["werewolf"])
        return _all, _live, _werewolf

    def _get_next_alive(self, crt: int) -> int:         # find in self._day_order, crt is self._next_player_idx
        number_of_players = len(self._characters)
        for idx in range(crt + 1, crt + number_of_players):
            current_idx = idx % number_of_players
            if self._is_alive[self._day_order[current_idx]]:
                return current_idx
        return crt

    def _text2vote(self, text, player_name=None) -> str:
        """
        convert text to vote, return a player's name
        """
        # lower = text.lower().replace("[", "").replace("]", "").replace(".", "")
        text = re.sub(r'\b(?:A|a)s(?:\s)?(?:the\s)?(?:P|p)layer(?:\s?[0-9]{1,2})?\b', '', text)
        text = re.sub(r'\b(?:I|i)\s(?:a|A|m|M)(?:\'?m|M)?\s(?:a|A|the|The)\s(?:P|p)layer(?:\s?[0-9]{1,2})?\b', '', text)
        text = re.split(r'[!.?:]', text)          # only save the last sentence to judge
        text_head = text[0] if text[0] != "" else text[1]
        text_tail = text[-1] if text[-1] != "" else text[-2]
        text_head = text_head.lower()
        if "myself" in text_head:
            assert player_name is not None
            return player_name
        for name in self.player_names:
            candidates = [name.lower(), name.lower().replace(" ", ""), name.lower().replace(" ", "_")]
            if any([candidate in text_head for candidate in candidates]):
                return name
        text_tail = text_tail.lower()
        if "myself" in text_head:
            assert player_name is not None
            return player_name
        for name in self.player_names:
            candidates = [name.lower(), name.lower().replace(" ", ""), name.lower().replace(" ", "_")]
            if any([candidate in text_tail for candidate in candidates]):
                return name
        return "pass"

    def get_rewards(self, chameleon_win: bool) -> Dict[str, float]:
        pass

    def is_terminal(self) -> bool:
        """
            check if the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content == SIGNAL_END_OF_CONVERSATION:
            return True

    def _kill_by_name(self, kill_list: List[str]):
        if '' in kill_list:
            kill_list.remove('')
        if "pass" in kill_list:
            kill_list.remove("pass")
        killed_identity = [self.player_names.index(name) for name in kill_list]
        for idx in killed_identity:
            if self._is_alive == False:
                continue
            self._is_alive[idx] = False
            self._alive_list.remove(self.player_names[idx])
            identity = self._characters[idx]
            self._identity_mapping[identity].remove(idx)
            '''if idx in self._night_order:
                self._night_order.remove(idx)'''
            self._lives -= 1

    def _judge_is_alive(self, name: str) -> bool:
        if name == "pass" or name == '':
            return False
        idx = self.player_names.index(name)
        return self._is_alive[idx]

    def _check_game_over(self):
        def _get_winner_names(is_villager=True):
            werewolf_camp = [self.player_names[idx] for idx, role in enumerate(self._characters) if role=="werewolf"]
            villager_camp = [name for name in self.player_names if name not in werewolf_camp]
            if is_villager:
                return villager_camp
            else:
                return werewolf_camp
        
        def _give_rewards(winner_names, camp):
            self.message_pool.give_rewards(winner_names)
            self.question_pool.give_rewards(last_turn=self._current_turn, camp=camp)
        
        if self._lives > 0 and len(self._identity_mapping["werewolf"]) == 0:
            self._moderator_speak("Game over, the villager wins!")
            _give_rewards(_get_winner_names(True), "villager")
            self._moderator_speak(SIGNAL_END_OF_CONVERSATION)
            return True
        # if self._current_phase == "night" and self._lives <= 2 and len(self._identity_mapping["werewolf"]) > 0:
        if len(self._identity_mapping["werewolf"]) > 0 and len(self._identity_mapping["villager"]) == 0:
            self._moderator_speak("Game over, the werewolf wins!")
            _give_rewards(_get_winner_names(False), "werewolf")
            self._moderator_speak(SIGNAL_END_OF_CONVERSATION)
            return True
        return False

    def step(self, player_name: str, action: str) -> TimeStep:
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        assert player_name == self.get_next_player(), f"Wrong player! It is {self.get_next_player()} turn."
        if self._current_phase == "daytime":
            assert self._get_number_of_people()[2] > 0
            rewards = self.get_zero_rewards()
            if self._number_of_rounds == 0:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                self._next_player_idx = self._get_next_alive(self._next_player_idx)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)

                print(f"_current_first_alive: {self._current_first_alive}", file=sys.stderr)
                if self._next_player_idx == self._current_first_alive:
                    self._next_player_idx = self._get_next_alive(-1)
                    print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                    self._number_of_rounds = 1
                    print(f"alive list: {self._alive_list}", file=sys.stderr)
                    self._moderator_speak(f"Now you {self.get_next_player()} are asked to choose which of the players should be voted for killing based on the discussion? Don't mention your role. "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. "
                                          f"For example: I vote to kill Player...")
                else:
                    self._moderator_speak(f"The next {self.get_next_player()}, you, "
                                          f"continue talking with other players based on your observation and reflection with few sentences. Decide whether to reveal your identity based on your reflection.",
                                          # f"For example: I observed that... I think that..."
                                          visible_to=self.get_next_player())
            elif self._number_of_rounds == 1:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                vote = self._text2vote(action, player_name)
                print(f"vote result: {vote}", file=sys.stderr)
                if vote in self.player_names or vote == "pass":
                    self._players_votes[vote] += 1
                self._next_player_idx = self._get_next_alive(self._next_player_idx)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)

                print(f"_current_first_alive: {self._current_first_alive}", file=sys.stderr)
                if self._next_player_idx == self._current_first_alive:
                    to_kill = max(self._players_votes, key=self._players_votes.get)
                    print(f"to kill: {to_kill}", file=sys.stderr)
                    print(f"is alive: {self._is_alive}", file=sys.stderr)
                    if to_kill != "pass" and not self._is_alive[self.player_names.index(to_kill)]:
                        self._moderator_speak(f"Only the living can be killed, {to_kill} is dead, "
                                              f"hence no one will be killed!")
                        to_kill = "pass"
                    else:
                        print(f"players votes: {str(self._players_votes)}", file=sys.stderr)
                        for name, vote in self._players_votes.items():
                            if name != to_kill and vote == self._players_votes[to_kill]:
                                to_kill = "pass"
                                self._moderator_speak("No consensus, no one will be killed!")
                                break
                    if to_kill == "pass":
                        self._current_turn += 1
                        self._number_of_rounds = 0
                        self._players_votes = {name: 0 for name in self.player_names}
                        self._players_votes["pass"] = 0
                        self._current_phase = "night"
                        # self._night_order = [x for k, v in self._identity_mapping if len(v) > 0 for x in v]
                        self._next_player_idx = 0
                        print(f"alive list: {self._alive_list}", file=sys.stderr)
                        self._moderator_speak(f"It's dark, everyone close your eyes.")
                        self._moderator_speak(f"Werewolves, please open your eyes! "
                                              f"Now vote and tell your teammates which of the players should be killed tonight. "
                                              f"You {self.get_next_player()} only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                              # f"For example: I choose Player...",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                              importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    else:
                        print(f"to kill: {to_kill}", file=sys.stderr)
                        self._moderator_speak(f"{to_kill} will be killed! You can make a brief last statement.", importance=6)
                        self._next_player_idx = self.player_names.index(to_kill)
                        self._number_of_rounds = 9999
                else:
                    self._moderator_speak(f"The next {self.get_next_player()}, you, continue voting the players should be killed based on the discussion? Don't mention your role. "
                                          f"Only choose one from the following living options please: [{', '.join(self._alive_list)}]. "
                                          f"For example: I vote to kill Player...",
                                          visible_to=self.get_next_player())
            else:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                self._print_infos()
                self._kill_by_name([player_name])
                self._print_infos()
                if self._check_game_over():
                    return TimeStep(observation=self.get_observation(), reward=rewards, terminal=True)
                self._current_turn += 1
                self._number_of_rounds = 0
                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0
                self._current_phase = "night"
                # self._night_order = [x for k, v in self._identity_mapping.items() if k != "villager" and len(v) > 0 for x in v]
                self._current_first_alive = self._get_next_alive(-1)
                self._next_player_idx = 0
                print(f"alive list: {self._alive_list}", file=sys.stderr)
                self._moderator_speak(f"It's dark, everyone close your eyes.")
                self._moderator_speak(f"Werewolves, please open your eyes! "
                                      f"Now vote and tell your teammates which of the players should be killed tonight. "
                                      f"You {self.get_next_player()} only choose one from the following living options please: [{', '.join(self._alive_list)}]. ", 
                                      # f"For example: I choose Player...",
                                      visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                      importance=1 if self._judge_is_alive(self.get_next_player()) else 0)

            terminal = False
            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=terminal)
        elif self._current_phase == "night":
            rewards = self.get_zero_rewards()
            # werewolf、guard、witch、seer
            # assert self._get_number_of_people()[2] > 0
            if self._next_player_idx < self.werewolf_numbers:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name)
                    print(f"vote result: {vote}", file=sys.stderr)
                    if vote in self.player_names or vote == "pass":
                        self._players_votes[vote] += 1
                self._next_player_idx += 1
                terminal = False

                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                print(f"number of wolves: {len(self._identity_mapping['werewolf'])}", file=sys.stderr)
                if self._next_player_idx == self.werewolf_numbers:
                    self._moderator_speak(f"You guard, {self.get_next_player()}, please open your eyes! "
                                          f"Now tell me who you protect tonight? "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to protect Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["guard"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    # self.print()
                else:
                    self._moderator_speak(f"The next werewolf, you {self.get_next_player()}, please vote and tell your teammates that which of the players should be killed tonight. "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
            elif self._next_player_idx == self.werewolf_numbers:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["guard"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name=player_name)
                    if "myself" in action:
                        vote = player_name
                    print(f"vote result: {vote}", file=sys.stderr)
                    if vote == '':
                        vote = "pass"
                    if self._guard_to == vote:
                        self._guard_to = "pass"
                    else:
                        self._guard_to = vote
                else:
                    self._guard_to = "pass"
                print(f"guard to: {self._guard_to}", file=sys.stderr)
                self._next_player_idx += 1
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                terminal = False

                werewolf_kill = max(self._players_votes, key=self._players_votes.get)
                print(f"werewolf_kill: {werewolf_kill}", file=sys.stderr)
                if self._judge_is_alive(werewolf_kill):
                    print(f"players votes: {str(self._players_votes)}", file=sys.stderr)
                    for name, vote in self._players_votes.items():
                        if name != werewolf_kill and vote == self._players_votes[werewolf_kill]:
                            werewolf_kill = "pass"
                            break
                    print(f"_guard_to: {self._guard_to}", file=sys.stderr)
                    if werewolf_kill == self._guard_to:
                        werewolf_kill = "pass"
                else:
                    werewolf_kill = "pass"
                if werewolf_kill != "pass":
                    self._print_infos()
                    self._killed_list.append(werewolf_kill)
                    self._print_infos()
                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0

                print(f"killed list: {self._killed_list}", file=sys.stderr)
                if len(self._killed_list) > 0 and self._witch_antidote:
                    self._moderator_speak(f"You witch, {self.get_next_player()}, please open your eyes! {self._killed_list[0]} will be killed tonight. "
                                          f"You have a bottle of antidote, do you want to save him? "
                                          f"Must choose only one from the following options: [Yes, No]",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                else:
                    print(f"alive list: {', '.join(self._alive_list)}", file=sys.stderr)
                    self._moderator_speak(f"You witch, {self.get_next_player()}, please open your eyes! "
                                          f"You have a bottle of poison, who are you going to kill tonight? "
                                          f"Choose one from the following living options: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to kill Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    self._number_of_rounds = 1
            elif self._next_player_idx == self.werewolf_numbers + 1:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._number_of_rounds == 0:
                    if self._judge_is_alive(player_name):
                        print(f"action: {action}", file=sys.stderr)
                        if "yes" in action or "Yes" in action or "will use" in action or "choose to use" in action:
                            if self._witch_antidote:
                                self._witch_antidote_to = self._killed_list[0]
                                self._witch_antidote = False
                                self._killed_list = []
                                print(f"_witch_antidote_to: {self._witch_antidote_to}", file=sys.stderr)
                                print(f"_witch_antidote: {self._witch_antidote}", file=sys.stderr)
                                print(f"_killed_list: {self._killed_list}", file=sys.stderr)
                            else:
                                self._moderator_speak("Failed, your antidote has run out!",
                                                      visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                                      importance=3)
                                self._witch_antidote_to = None

                    self._print_infos()
                    self._kill_by_name(self._killed_list)
                    if len(self._killed_list) != 0:
                        self._night_kill_list.append(self._killed_list[0])
                    self._killed_list = []
                    self._print_infos()
                    self._moderator_speak(f"You {self.get_next_player()} have a bottle of poison, who are you going to kill tonight? "
                                          f"Choose only one from the following living options: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to kill Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    self._number_of_rounds = 1
                elif self._number_of_rounds == 1:
                    if self._judge_is_alive(player_name):
                        vote = self._text2vote(action, player_name=player_name)
                        print(f"vote result: {vote}", file=sys.stderr)
                        if self._witch_poison:
                            self._witch_poison_to = vote if self._judge_is_alive(vote) else "pass"
                            self._witch_poison = False if self._judge_is_alive(vote) else True
                            print(f"_witch_poison_to: {self._witch_poison_to}", file=sys.stderr)
                            print(f"_witch_poison: {self._witch_poison}", file=sys.stderr)
                        else:
                            self._moderator_speak("Failed, your poison has run out!",
                                                  visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                                  importance=3)
                            self._witch_poison_to = None
                        if self._witch_poison_to is not None and self._witch_poison_to != "pass":
                            if self._witch_poison_to not in self._killed_list:
                                self._killed_list.append(self._witch_poison_to)
                    self._print_infos()
                    self._kill_by_name(self._killed_list)
                    if len(self._killed_list) != 0:
                        self._night_kill_list.append(self._killed_list[0])
                    self._killed_list = []
                    self._print_infos()
                    self._next_player_idx += 1
                    self._number_of_rounds = 0

                    print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                    self._moderator_speak(f"You seer, {self.get_next_player()}, please open your eyes! "
                                          f"Who are you going to verify its identity tonight? "
                                          f"Choose only one from the following living options: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to verify Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)

                terminal = False
            elif self._next_player_idx == self.werewolf_numbers + 2:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name)
                    print(f"vote result: {vote}", file=sys.stderr)
                    idx = 0 if vote == "pass" else self.player_names.index(vote)
                    print(f"idx: {idx}", file=sys.stderr)
                    if self._characters[idx] == "werewolf":
                        self._moderator_speak(f"{self.player_names[idx]} is a werewolf!",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                              importance=5)
                    else:
                        self._moderator_speak(f"{self.player_names[idx]} is not a werewolf!",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                              importance=5)

                self._next_player_idx += 1
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                terminal = False

                self._moderator_speak("The sun rose. Everyone woke up except those who had been killed.")
                print(f"killed list: {self._night_kill_list}", file=sys.stderr)
                if len(self._night_kill_list) != 0:
                    self._moderator_speak(f"{','.join(self._night_kill_list)} died last night!", importance=6)
                    if self._number_of_nights == 0:
                        killed_identity = [self._characters[self.player_names.index(name)] for name in self._night_kill_list]
                        print(f"killed_identity: {killed_identity}", file=sys.stderr)
                        self._moderator_speak(f"{', '.join(self._night_kill_list)} are {', '.join(killed_identity)}.", importance=5)
                else:
                    self._moderator_speak("It was a peaceful night and no one died!", importance=4)

                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0
                self._guard_to = None
                self._witch_antidote_to = None
                self._witch_poison_to = None
                self._killed_list = []
                self._night_kill_list = []
                if self._check_game_over():
                    return TimeStep(observation=self.get_observation(), reward=rewards, terminal=True)
                self._next_player_idx = self._current_first_alive = self._get_next_alive(-1)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                self._current_phase = "daytime"
                
                self._moderator_speak(f"Now freely talk about roles of other players with each other based on your observation and "
                                      f"reflection with few sentences. Decide whether to reveal your identity based on your reflection. The first {self.get_next_player()}, you please.\n"
                                      # f"For example: I observed that... I think that..."
                                      )

                self._number_of_nights += 1
                
            else:
                raise ValueError(f"Unknown player_idx: {self._next_player_idx}")

            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=terminal)
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
