from typing import List
import os
import re
import random
import logging
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
from requests.auth import HTTPBasicAuth
import sys

from .base import IntelligenceBackend
from ..message import Message
from ..message import Message, MessagePool, Question, QuestionPool

try:
    import openai
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if openai.api_key is None:
        logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False
    else:
        is_openai_available = True

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-3.5-turbo-0301"

STOP = ("<EOS>", "[EOS]", "(EOS)")  # End of sentence token
END_OF_MESSAGE = "<EOS>"


class OpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"

    def __init__(self, args, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, **kwargs):
        if not args or (args and not args.use_api_server):
            assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(args, temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)

        if args:
            self.temperature = args.temperature
        else:
            self.temperature = temperature
        if args:
            self.max_tokens = args.max_tokens
        else:
            self.max_tokens = max_tokens
        self.model = model

    @retry(stop=stop_after_attempt(10), wait=wait_random_exponential(min=60, max=120))
    def _get_response(self, messages, method, max_tokens=None, T=None):
        max_toks = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if T is None else T
        if method == 0:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_toks,
                stop=STOP
            )
            response = completion.choices[0]['message']['content']
        if method == 1:
            # Warning!!! If you use your self-constructed API server, you should configure it here.
            url = "http://***********"
            data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_toks,
                    "stop": STOP
                }
            data = json.dumps(data)
            print(f"  Temperature: {temperature}, Max_tokens: {max_toks}", file=sys.stderr)
            completion = requests.post(url=url, data=data, auth=HTTPBasicAuth(username="****",password="****")).json()
            # print(completion)
            response = completion['choices'][0]['message']['content']
        
        response = response.strip()
        return response
    
    def extract_text(self, s):
        patterns = [
            r': "(.+?)"', 
            r'content: (.+)', 
            r'content:\n(.+)', 
            r'content:\n\n(.+)', 
            r'content: \n(.+)',
            r'night: (.+)'
            r'night:\n(.+)', 
            r'night:\n\n(.+)', 
            r'night: \n(.+)',
            r'daytime: (.+)'
            r'daytime:\n(.+)', 
            r'daytime:\n\n(.+)', 
            r'daytime: \n(.+)',
            r'"(.+)"', 
            r'"(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, s)
            if match:
                return match.group(1)
        
        return s

    def query(self, arg, agent_name: str, role_desc: str, history_messages: List[Message], msgs: MessagePool, ques: QuestionPool, global_prompt: str = None,
              request_msg: Message = None, turns = 0, day_night = "daytime", role="", alives=[], *args, **kwargs) -> str:
        """
        format the input and call the ChatGPT/GPT-4 API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """
        def _get_branch(task, day_night, role):
            if "Choose" in task["content"] or "choose" in task["content"] or "vote to" in task["content"] or "Yes, No" in task["content"]:
                if role == "werewolf" or role == "seer" or role == "guard":
                    if day_night == "night":
                        return 2
                if role == "witch" and day_night == "night" and "Yes" in task["content"]:
                    return 2
                if role == "witch" and day_night == "night":
                    return 3
                return 1
            else:
                return 0
        
        conn_method = arg.use_api_server if arg and arg.use_api_server else 0
        max_tokens = arg.max_tokens if arg and arg.max_tokens else 100
        temperature = arg.temperature if arg and arg.temperature else 0.2
        alives = alives.copy()
        alives.remove('pass')
        alive_players = "Living players now: " + ", ".join(alives) + "."
        if arg:
            f = open(os.path.join(arg.logs_path_to, str(arg.current_game_number) + ".md"), "a")
            f.write(f"**{agent_name}**:  \n")

        conversations = []
        for i, message in enumerate(history_messages):
            if message.agent_name == agent_name:
                conversations.append({"role": "assistant", "content": f"{message.agent_name}: {message.content}{END_OF_MESSAGE}"})
            else:
                # Since there are more than one player, we need to distinguish between the players
                conversations.append({"role": "user", "content": f"{message.agent_name}: {message.content}{END_OF_MESSAGE}"})

        global_desc = f"The following is the chat history you observed. You are {agent_name}, the {role}."
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt_str = f"{global_prompt.strip()}\n{role_desc}\n{global_desc}"
        else:
            system_prompt_str = role_desc
        system_prompt = {"role": "system", "content": system_prompt_str}
        
        question_list = ques.get_necessary_questions()
        if day_night == "daytime":
            question_list.append("Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?")
        initial_question = '\n'.join(question_list)
        if arg and arg.use_crossgame_ques and role in arg.who_use_ques:
            retrieve_list = ques.get_best_questions(role, arg.retri_question_number, use_history=True)
        else:
            retrieve_list = ques.get_best_questions(role, arg.retri_question_number)
        retrieve_list = [que.content for que in retrieve_list]
        retrieve_question = '\n'.join(retrieve_list)
        request_prompt = [{"role": "system", "content": f"Now its the {turns}-th {day_night}. Given the game rules and conversations above, assuming you are {agent_name}, the {role}, and "
                            f"to complete the instructions of the moderator, you need to think about a few questions clearly first, so that you can make an accurate decision on the next step.\n"
                            f"{initial_question}\n\nDo not select or answer the questions above. Except for the question above, choose only three that you think are the most important in the current situation from the list of questions below:\n\n"
                            f"{retrieve_question}\n\nPlease repeat the three important questions of your choice, separating them with '#'.{END_OF_MESSAGE}"
                        }]
        request = [system_prompt] + conversations + request_prompt
        print(f"request: {request}", file=sys.stderr)
        response = self._get_response(request, conn_method, max_tokens=150, T=temperature, *args, **kwargs)
        print(f"response: {response}", file=sys.stderr)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        # print(response)
        selected_list = [s.strip() for s in response.split('#') if s.strip() != ""]
        selected_question = '\n'.join(selected_list)
        request_prompt = [{"role": "system", "content": f"Now its the {turns}-th {day_night}. Given the game rules and conversations above, assuming you are {agent_name}, the {role}, and "
                            f"to complete the instructions of the moderator, you need to think about a few questions clearly first, so that you can make an accurate decision on the next step.\n\n"
                            f"{initial_question}\n{selected_question}\n\nDo not answer these queations. In addition to the above questions, please make a bold guess, "
                            f"what else do you want to know about the current situation? Please ask two important questions in first person, separating them with '#'.{END_OF_MESSAGE}"
                        }]
        request = [system_prompt] + conversations + request_prompt
        print(f"request: {request}", file=sys.stderr)
        response = self._get_response(request, conn_method, max_tokens=150, T=0.8, *args, **kwargs)
        response = re.sub(r'\d\.\s', '', response)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        print(f"response: {response}", file=sys.stderr)
        ask_list = [s.strip() for s in response.split('#') if s.strip() != ""]
        for que in ask_list:
            ques.append_question(Question(que, turn=arg.current_game_number, visible_to=role, reward=500))
        final_questions = (question_list + selected_list + ask_list)
        final_questions.append("Based on the conversation above, which players have clearly implied their roles?")
        
        request_prompt = []
        q_a = []
        topk = arg.answer_topk if arg and arg.answer_topk else 5
        for i, question in enumerate(final_questions):
            question = question.strip()
            if question == "":
                continue
            # Ask the question to LLM for more better answer!
            if i != len(final_questions) - 1:
                answers = msgs.find_k_most_similar(agent_name, question, topk)
                len_answers = min(topk, len(answers))
                answers = ['<' + str(i + 1) + '> ' + answer.strip() for i, answer in enumerate(answers)]
                answers = '\n'.join(answers)
                request = [{
                    "role": "system", "content": f"Now its the {turns}-th {day_night}. Given the game rules and conversations above, assuming you are {agent_name}, the {role}, for question:\n{question}\n\nThere are {len_answers} possible answers:\n{answers}\n\n"
                    f"Generate the correct answer based on the context. If there is not direct answer, you should think and "
                    f"generate your answer based on the context. No need to give options. The answer should in first person using no more than 2 sentences "
                    f"and without any analysis and item numbers.{END_OF_MESSAGE}"
                }]
            else:
                request = [{
                    "role": "system", "content": f"Now its the {turns}-th {day_night}. Assuming you are {agent_name}, the {role}, {question} "
                    f"Only generate the player name and its possible role based on the context. If there is no clue, generate 'No identity clues revealed.'. "
                    f"The answer should in first person using no more than 3 sentences.{END_OF_MESSAGE}"
                }]
            request = [system_prompt] + conversations + request_prompt + request
            print(f"Ask to LLM for answer: {request}", file=sys.stderr)
            answer = self._get_response(request, conn_method, T=0.1, *args, **kwargs)
            answer = re.sub(rf"{END_OF_MESSAGE}$", "", answer).strip()
            print(f"Answer from LLM: {answer}", file=sys.stderr)
            answer = answer.replace('\n', ' ').strip()
            if i == 0:
                request_prompt.append({"role": "assistant", "content": f"Current inner thinking in my heart (not happened):\n\n{alive_players}\n{answer}{END_OF_MESSAGE}"})
            else:
                request_prompt[-1]["content"] = f"{request_prompt[-1]['content'][:-5]}\n{answer}{END_OF_MESSAGE}"
            q_a.append(question + ' ' + answer)
        
        if arg:
            q_a = '\n'.join(q_a)
            f.write(f"- **Q&A**: {q_a}  \n")
            
        # task = conversations.pop()
        request = [system_prompt] + conversations + request_prompt
        request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Assuming you are {agent_name}, the {role}, what insights "
                        f"can you summarize with few sentences based on the above conversations and current inner thinking in heart "
                        f"for helping continue the talking and achieving your objective? "
                        f"Example: As the {role}, I observed that... I think that... But I am... So...{END_OF_MESSAGE}"
                        })
        print(f"request2: {request}", file=sys.stderr)
        response = self._get_response(request, conn_method, max_tokens=200, *args, **kwargs)
        print(f"response2: {response}", file=sys.stderr)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        
        reflexions = response.replace('\n', ' ')
        if arg:
            f.write(f"- **Reflexion**: {reflexions}  \n")
            
        reflexions = {"role": "assistant", "content": f"My reflection in heart (not happened): {reflexions}{END_OF_MESSAGE}"}
        ref_new = Message(agent_name, reflexions["content"].replace(END_OF_MESSAGE, ''), turn=msgs.last_turn, visible_to=agent_name, msg_type="ref")
        msgs.append_message(ref_new)
        task = conversations.pop()
        task["role"] = "user"
        
        branch = _get_branch(task, day_night, role)
        
        if arg and arg.use_crossgame_exps and role in arg.who_use_exps:
            if arg.exps_retrieval_threshold:
                print("################################ To Retrieve experiences!", file=sys.stderr)
                print(f"role: {role}, branch: {branch}", file=sys.stderr)
                exps = msgs.get_best_experience(reflexions["content"].split(': ', maxsplit=1)[1], role, branch, threshold=arg.exps_retrieval_threshold)
            else:
                exps = msgs.get_best_experience(reflexions["content"].split(': ', maxsplit=1)[1], role, branch)
        else:
            exps = None
        
        
        if exps is None:
            if arg:
                f.write(f"- **Exps**: None  \n")
            request = [system_prompt] + conversations + [reflexions] + [task]
            if "Choose" in task["content"] or "choose" in task["content"] or "vote to" in task["content"] or "Yes, No" in task["content"]:
                request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Think about which to choose based on the context, especially the just now reflection. "
                                "Tip: you should kill/save the player as soon as possible once you have found the player is your enemy/teammate. "
                                "Give your step-by-step thought process and your derived consise talking content (no more than 2 sentences) at last. For example: My step-by-step thought process:... My concise talking content: I choose..."})
                Temp = 0.0
            else:
                request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Think about what to say in your talking based on the context. Give your step-by-step thought process and your derived consise talking content at last.  For example: My step-by-step thought process:... My concise talking content:..."})
                '''request.append({"role": "system", "content": f"Combining the conversations, reflections above, assuming you are {agent_name}, the {role}, "
                                f"continue to talk with few concise sentences. You'd better not reveal your role, because there may be your enemies in other players.{END_OF_MESSAGE}"})'''
                Temp = arg.temperature
        else:
            request = [system_prompt]
            good_exps = '\n'.join(exps[0])
            if "Choose" in task["content"] or "choose" in task["content"] or "vote to" in task["content"] or "Yes, No" in task["content"]:
                request.append({'role': 'user', 'content': f"I retrieve some historical experience similar to current situation that I am facing. "
                                f"There is one bad experience:\n\n{exps[1]}\n\nAnd there are also a set of experience that may consist of good ones:\n\n{good_exps}\n\n"
                                "Please help me analyze the differences between these experiences and identify the good ones from the set of experiences. "
                                "The difference is mainly about voting to kill someone or to pass, choosing to protect someone or to pass, using drugs or not. "
                                "What does the experience set do but the bad experience does not do? "
                                "Indicate in second person what is the best way for the player to do under such reflection. Clearly indicate whether to vote, protect or use drugs without any prerequisites. "
                                "For example 1: The experience set involves choosing to protect someone, while the bad experience involves not protecting anyone and choosing to pass in contrast. "
                                "The best way for you to do under such reflection is to choose someone to protect based on your analysis.\n"
                                "For example 2: The bad experience choose to pass the voting, and all the experience in the experience set choose to pass as well. "
                                "The best way for you to do under such reflection is to observe and analyse the identity of other players.\n"
                                "No more than 1 sentence. If there is no obvious difference between them, only generate 'No useful experience can be used.'.<EOS>"})
            else:
                request.append({'role': 'user', 'content': f"I retrieve some historical experience similar to current situation that I am facing. "
                            f"There is one bad experience:\n\n{exps[1]}\n\nAnd there are also a set of experience that may consist of good ones:\n\n{good_exps}\n\n"
                            "According to the game result, good experience may be better than bad experience and lead game victory faster than bad experience. "
                            "Compare and find the difference between the bad experience and the experience set, this is the key to victory. Ignore the player name and think what good experience set do but bad experience not do and "
                            "do not say to me. Indicate in second person what is the best way for the player to do under such reflection? For example: The best "
                            "way for you to do under such reflection is to...\nNo more than 1 sentence. If there is no obvious difference between them, only "
                            "generate 'No useful experience can be used.'.<EOS>"})
            print(f"request2: {request}", file=sys.stderr)
            response = self._get_response(request, conn_method, max_tokens=200, *args, **kwargs)
            print(f"response2: {response}", file=sys.stderr)
            response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
            if re.search('The best way.*', response):
                response = re.search('The best way.*', response).group()
            response = re.sub(r"(\sor.*)(\.)", r'\2', response)
            exp = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
            
            request = [system_prompt] + conversations + [reflexions] + [task]
            if "Choose" in task["content"] or "choose" in task["content"] or "vote to" in task["content"]:
                request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Think about which to choose based on the context, especially the just now reflection. "
                                f"Besides, there maybe history experience you can refer to: {exp} Give your step-by-step thought process and your derived consise talking content (no more than 2 sentences) at last. "
                                "For example: My step-by-step thought process:... My concise talking content: I choose..."})
                Temp = 0.0
            elif "Yes, No" in task["content"]:
                request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Think about which to choose based on the context, especially the just now reflection. "
                                f"Besides, there maybe history experience you can refer to: {exp} Give your step-by-step thought process and your derived consise talking content (no more than 2 sentences) at last. "
                                "For example: My step-by-step thought process:... My concise talking content:..."})
                Temp = 0.0
            else:
                request.append({"role": "system", "content": f"Now its the {turns}-th {day_night}. Think about what to say in your talking based on the context. "
                                f"Besides, there maybe history experience you can refer to: {exp} Give your step-by-step thought process and your derived consise talking content at last. "
                                "For example: My step-by-step thought process:... My concise talking content:..."})
                Temp = arg.temperature
            
            if arg:
                f.write(f"- **Exps**: {exp.strip()} \n")
        print(f"request: {request}", file=sys.stderr)
        response = self._get_response(request, conn_method, T=Temp, max_tokens=400, *args, **kwargs)
        print(f"raw response: {response}", file=sys.stderr)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        if arg:
            f.write(f"- **CoT**: {response}  \n\n")
        # print(response)
        response = self.extract_text(response)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        # print(response)
        response = response.replace('\n', ' ')
        response = response.replace("'''.", '')
        response = response.strip('"')
        response = response.strip("'")
        print(f"response: {response}", file=sys.stderr)
        
        game_number = arg.current_game_number if arg and arg.current_game_number else 0
        exp_new = Message(agent_name, [reflexions["content"].replace(END_OF_MESSAGE, '').split(': ', maxsplit=1)[1], response, 0, branch], turn=game_number, msg_type="exp")
        msgs.append_message(exp_new)
        
        if "Choose" in task["content"] or "choose" in task["content"] or "vote to" in task["content"] or "Yes, No" in task["content"]:
            response = f"({turns}-th {day_night}) " + response
        
        if arg:
            f.write(f"- **Final**: {response}  \n\n")
            f.close()
        
        return response
