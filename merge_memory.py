from chatarena.message import Message, MessagePool

import sys
import os
import pickle
from string import ascii_uppercase
import re

b_from = int(sys.argv[1])
b_to = int(sys.argv[2])

file_names = ["exps_" + str(i) + "_nonincremental.pkl" for i in range(b_from, b_to + 1)]

messages = []

for file in file_names:
    
    print(f"{file}")
    with open(os.path.join("./exps", file), "rb") as f:
        msgs = pickle.load(f)
        messages.extend(msgs)
        
def replace_player_numbers(s1, s2):
    
    nums_s1 = set(int(num) for num in re.findall(r'Player (\d+)', s1))
    nums_s2 = set(int(num) for num in re.findall(r'Player (\d+)', s2))
    nums = sorted(nums_s1.union(nums_s2))

    num_to_letter = {num: f'<{letter}>' for num, letter in zip(nums, ascii_uppercase)}

    for num, letter in num_to_letter.items():
        s1 = s1.replace(f'Player {num}', f'Player {letter}')
        s2 = s2.replace(f'Player {num}', f'Player {letter}')

    return s1, s2

print("Masking experiences...")
result_msgs = []
for msg in messages:
    assert msg.msg_type == "exp"
    reflexion = msg.content[0]
    action = msg.content[1]
    new_msg = msg
    
    reflexion_after, action_after = replace_player_numbers(reflexion, action)
    new_msg.content[0], new_msg.content[1] = reflexion_after, action_after
    result_msgs.append(new_msg)

file_name = "exps_" + str(b_from) + "_" + str(b_to) + ".pkl"
with open(os.path.join("./exps", file_name), "wb") as f:
    pickle.dump(result_msgs, f)
print("Done!")
