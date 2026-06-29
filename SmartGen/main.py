import argparse
import pickle
import ast
import os
import json
import time
from openai import OpenAI
from dictionary import dayofweek_dict, hour_dict, fr_devices_dict, fr_actions, sp_devices_dict, sp_actions, us_devices_dict, us_actions
from split import Split
from dayse import Dayse
from transtext import Transtext
from transnumber import Transnum
from sppc import SPPC_select, similarity_select
from baseline1 import Anomaly_detection
from baseline2 import Train
from text_translation_matrix import ATM
from extract import Extract
from find_categories import Find_categories
from security_check import security_check
from dotenv import load_dotenv
load_dotenv()
#from SAS_main import SASRec_behavior_prediction

vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
device_dic = {"us": us_devices_dict, "fr": fr_devices_dict, "sp": sp_devices_dict}
act_dic = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}

BATCH_SIZE = 35  # 1728 total seqs / (7 keys x 20 req) = 12.3; 90 keeps each batch a reasonable size
BATCH_DELAY = 5   # short pause between successful batches (seconds)

API_KEYS = [k for k in [
    os.getenv("Grok_API2"),
    os.getenv("Grok_API3"),
    os.getenv("Grok_API4"),
    os.getenv("Grok_API5"),
    os.getenv("Grok_API6"),
    os.getenv(Grok_API7"),
    os.getenv("Grok_API8"),
] if k]  # silently skip any unset keys

current_key_index = 0  # global pointer; rotated automatically on every 429 error


def get_client(key_index):
    """Create an OpenAI-compatible Gemini client for the given key index."""
    return OpenAI(
        api_key=API_KEYS[key_index],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def get_args_parser():
    parser = argparse.ArgumentParser('LLM generation', add_help=False)
    parser.add_argument('--model', default='gpt-4o', type=str,
                        help='The used LLM: Llama_405B/70B/gpt-4o/deepseek-v3')
    parser.add_argument('--dataset', default='fr', type=str,
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--ori_env', default='winter', type=str,
                        help='The original home environment: winter/daytime')
    parser.add_argument('--new_env', default='spring', type=str,
                        help='The new home environment: spring/night')
    parser.add_argument('--method', default='SPPC', type=str,
                        help='The compression method: SPPC/similarity/instance')
    parser.add_argument('--threshold', default=0.918, type=float,
                        help="The compression threshold")
    parser.add_argument('--percentage', default=95.5, type=float,
                        help='The anomaly detection threshold percentage')
    parser.add_argument('--need_test', default=True, type=bool,
                        help='The experimental setup: True/False')
    parser.add_argument('--need_generate', default=False, type=bool,
                        help='The experimental setup: True/False')
    return parser


def LLM_call(prompt):
    """
    Call the LLM with automatic key rotation on 429 errors.
    Tries every available key before giving up.
    """
    global current_key_index

    for attempt in range(len(API_KEYS)):
        try:
            client = get_client(current_key_index)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                stream=False,
                messages=messages,
                max_tokens=8040,
                temperature=0,
                top_p=0,
            )
            result = response.choices[0].message.content.strip()
            print(result)
            return result

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"  [Key {current_key_index + 1}/{len(API_KEYS)}] Quota exhausted. Rotating to next key...")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                if attempt < len(API_KEYS) - 1:
                    time.sleep(3)  # brief pause before retrying with new key
                    continue
            raise  # re-raise non-quota errors immediately

    raise RuntimeError("All API keys exhausted their daily quota. Try again tomorrow.")


def build_prompt(sentence, device_control_dict, action_transition, batch):
    """Build the LLM prompt for a single batch of sequences."""
    return (
        "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. "
        "Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. "
        "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. "
        "And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences."
        "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices."
        f"The set of the possible device and device states: {device_control_dict}"
        f"{sentence} The user's compressed original sequences of behavior: {batch}. User's behavior habits: {action_transition}"
        "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. "
        "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits."
        "Requirements:"
        "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices."
        "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device."
        "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment."
        "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors."
        "5.Please also generate reasonable behavior time when generating, not just a single behavior."
        "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>"
        "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats."
        "Please think step by step, and return the final generated user behavior sequence set."
    )


def extract_sequences_from_response(response):
    """
    Parse the <seq [...] seq> block out of an LLM response.
    Returns a Python list of subsequences, or [] on failure.
    """
    try:
        start = response.index("<seq") + len("<seq")
        end = response.index("seq>", start)
        raw = response[start:end].strip()
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        print(f"[WARNING] Failed to parse LLM response: {e}")
        return []


def merge_sequences(all_sequences):
    """Wrap a flat list of subsequences back into the <seq [...] seq> string format."""
    return f"<seq {all_sequences} seq>"


def call_llm_in_batches(user_sequence, sentence, device_control_dict, action_transition):
    """
    Split user_sequence into BATCH_SIZE chunks, call the LLM for each (with
    automatic key rotation on quota errors), parse results, and return one
    merged <seq [...] seq> string.
    """
    merged = []
    total_batches = (len(user_sequence) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(0, len(user_sequence), BATCH_SIZE):
        batch = user_sequence[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"  [Batch {batch_num}/{total_batches}] sequences {batch_start}–{batch_start + len(batch) - 1} | using key {current_key_index + 1}/{len(API_KEYS)}")

        prompt = build_prompt(sentence, device_control_dict, action_transition, batch)
        response = LLM_call(prompt)

        parsed = extract_sequences_from_response(response)
        if parsed:
            merged.extend(parsed)
        else:
            print(f"  [Batch {batch_num}] No valid sequences extracted; skipping.")

        if batch_start + BATCH_SIZE < len(user_sequence):
            time.sleep(BATCH_DELAY)

    return merge_sequences(merged)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)

    if not API_KEYS:
        raise RuntimeError("No API keys found. Set gemini_API_1 ... gemini_API_7 in your .env file.")
    print(f"Loaded {len(API_KEYS)} API key(s).")

    if args.need_generate:
        Split(args.dataset, args.ori_env, 1)
        Dayse(args.dataset, args.ori_env)
        if args.method == 'SPPC':
            Train(args.dataset, args.ori_env, vocab_dic[args.dataset])
            SPPC_select(args.dataset, args.ori_env, vocab_dic[args.dataset], args.threshold)
        elif args.method == 'similarity':
            similarity_select(args.dataset, args.ori_env, args.threshold)

        all_categories = Find_categories(args.dataset, args.ori_env, args.method, args.threshold)
        device_dict = device_dic[args.dataset]
        actions = act_dic[args.dataset]
        dictionaries = [dayofweek_dict, hour_dict, device_dict, actions]

        ATM(args.dataset, args.ori_env, actions)
        Transtext(args.dataset, args.ori_env, args.threshold, args.method, all_categories, dictionaries)

        if args.new_env == 'spring':
            sentence = f'The previous environment is {args.ori_env}. The changed environment is warm {args.new_env}.'
        elif args.new_env == 'night':
            sentence = (
                f'The previous environment: user is active during the {args.ori_env} and rest at {args.new_env}. '
                f'The changed environment: user is active at {args.new_env} and rest during the {args.ori_env}.'
            )
        elif args.new_env == 'multiple':
            sentence = (
                f'The previous environment was for a {args.ori_env} person to be at home, '
                f'and the changed environment is for {args.new_env} people to be at home'
            )

        with open(f'{args.dataset}_keys_best.txt', 'r') as file:
            device_control_dict = file.read()

        with open(f'IoT_data/{args.dataset}/{args.ori_env}/action_transitions.json', 'r', encoding='utf-8') as f:
            action_transition = json.load(f)

        for day in all_categories:
            print(f"\n=== Processing day category: {day} ===")
            with open(
                f'IoT_data/{args.dataset}/{args.ori_env}/trn_day_{day}_{args.method}_th={args.threshold}_text.pkl',
                'rb'
            ) as file3:
                user_sequence = pickle.load(file3)
                print(f"Total sequences for day {day}: {len(user_sequence)}")

            combined_response = call_llm_in_batches(
                user_sequence, sentence, device_control_dict, action_transition
            )

            out_path = (
                f'IoT_data/{args.dataset}/{args.new_env}/'
                f'{args.dataset}_{args.new_env}_generation_day_{day}_{args.method}_th={args.threshold}_{args.model}.pkl'
            )
            with open(out_path, 'wb') as f3:
                pickle.dump(combined_response, f3)
            print(f"Saved merged response to {out_path}")

        Extract(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories)
        Transnum(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories, dictionaries)
        security_check(args.dataset, args.new_env, args.threshold, args.method, args.model)

    if args.need_test:
        Anomaly_detection(args.dataset, args.new_env, args.threshold, args.method, args.model, args.percentage)
        # SASRec_behavior_prediction(args.dataset, args.new_env, args.threshold, args.method, args.model, need='train')
        # SASRec_behavior_prediction(args.dataset, args.new_env, args.threshold, args.method, args.model, need='test')
