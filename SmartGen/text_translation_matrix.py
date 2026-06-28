# import json
# import pickle
# from typing import List, Tuple, Dict

# import numpy as np


# class LinkAnalyzer:
#     def __init__(self, actions: Dict[str, int]):
#         self.actions = actions
#         self.number_set = set()
#         self.transition_matrix = None
#         self.number_to_index = {}
#         self.index_to_number = {}
#         self.link_counts = {}

#         for action, index in actions.items():
#             self.index_to_number[index] = action

#     def fit_sequences(self, sequences: List[List[int]]):

#         for numbers in sequences:
#             self.number_set.update(numbers)

#         for idx, num in enumerate(sorted(self.number_set)):
#             self.number_to_index[num] = idx
#             self.link_counts[num] = {}

#         for numbers in sequences:
#             for i in range(len(numbers) - 1):
#                 current_num = numbers[i]
#                 next_num = numbers[i + 1]

#                 if next_num not in self.link_counts[current_num]:
#                     self.link_counts[current_num][next_num] = 0
#                 self.link_counts[current_num][next_num] += 1

#     def get_top_transitions(self, top_n: int = 5) -> Dict[int, List[Tuple[int, int]]]:
#         top_transitions = {}
#         for num, transitions in self.link_counts.items():
#             sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:top_n]
#             top_transitions[num] = sorted_transitions
#         return top_transitions

#     def print_transition_matrix(self):
#         num_count = len(self.number_set)
#         self.transition_matrix = np.zeros((num_count, num_count), dtype=int)

#         for current_num, transitions in self.link_counts.items():
#             current_idx = self.number_to_index[current_num]
#             for next_num, count in transitions.items():
#                 next_idx = self.number_to_index[next_num]
#                 self.transition_matrix[current_idx][next_idx] = count

#         print("Link matrix (rows represent current action, list represents next action): ")
#         print("  ", end="")
#         for num in sorted(self.number_set):
#             print(f"{self.index_to_number[num]:>30}", end=" ")
#         print()

#         for i, num in enumerate(sorted(self.number_set)):
#             print(f"{self.index_to_number[num]:>30} ", end="")
#             for j in range(num_count):
#                 print(f"{self.transition_matrix[i][j]:>3}", end=" ")
#             print()


# def analyze_link(sequences: List[List[int]], actions: Dict[str, int], file_name):
#     analyzer = LinkAnalyzer(actions)
#     analyzer.fit_sequences(sequences)

#     top_transitions = analyzer.get_top_transitions(top_n=5)

#     print("The top 5 most commonly followed actions after each action: ")
#     for num, transitions in top_transitions.items():
#         if not transitions:
#             print(f"Actions {analyzer.index_to_number[num]} are usually not followed by any other actions")
#         else:
#             print(f"Actions {analyzer.index_to_number[num]} most common action afterwards: ")
#             for next_num, count in transitions:
#                 print(f"  -> {analyzer.index_to_number[next_num]}: {count} times")

#     transition_results = {}

#     for num, transitions in top_transitions.items():
#         action_key = str(analyzer.index_to_number[num])

#         if not transitions:
#             transition_results[action_key] = {
#                 "message": "It is usually not followed by the next action.",

#             }
#         else:
#             transition_data = []
#             for next_num, count in transitions:
#                 transition_data.append({
#                     "next_action": analyzer.index_to_number[next_num],
#                     "count": count
#                 })

#             transition_results[action_key] = {
#                 "message": "The most common action following it",
#                 "transitions": transition_data
#             }

#     with open(file_name, 'w', encoding='utf-8') as f:
#         json.dump(transition_results, f, ensure_ascii=False, indent=4)

#     analyzer.print_transition_matrix()


# def ATM(dataset, ori_env, actions):
#     with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'rb') as file2:
#         data = pickle.load(file2)

#     indices_to_extract = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

#     sequences = [
#         [sublist[i] for i in indices_to_extract if i < len(sublist)]
#         for sublist in data
#     ]
#     file_name = f'IoT_data/{dataset}/{ori_env}/action_transitions.json'
#     analyze_link(sequences, actions, file_name)

#     with open(file_name, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         print(json.dumps(data, ensure_ascii=False, indent=2))


import json
import math
import pickle
from typing import List, Tuple, Dict


class LinkAnalyzer:
    def __init__(self, actions: Dict[str, int]):
        self.actions = actions
        self.index_to_action = {v: k for k, v in actions.items()}
        self.link_counts: Dict[int, Dict[int, int]] = {}
        self.number_set: set = set()
        self.total_sequences: int = 0

    def fit_sequences(self, sequences: List[List[int]]):
        self.total_sequences = len(sequences)

        for seq in sequences:
            self.number_set.update(seq)

        for num in self.number_set:
            self.link_counts[num] = {}

        for seq in sequences:
            for i in range(len(seq) - 1):
                src, dst = seq[i], seq[i + 1]
                self.link_counts[src][dst] = self.link_counts[src].get(dst, 0) + 1



    @staticmethod
    def otsu_threshold(values: List[float], bins: int = 20) -> float:
    
        if len(values) < 3:
            return 0.0

        mn, mx = min(values), max(values)
        if mn == mx:
            return mn

        n_bins = min(bins, len(values))
        step = (mx - mn) / n_bins

        counts = [0] * n_bins
        for v in values:
            idx = min(int((v - mn) / step), n_bins - 1)
            counts[idx] += 1

        total = len(values)
        total_mean = sum(
            counts[i] * (mn + (i + 0.5) * step)
            for i in range(n_bins)
        ) / total

        best_thresh = mn
        best_var    = -1.0
        w0, sum0    = 0.0, 0.0

        for i in range(n_bins - 1):
            mid  = mn + (i + 0.5) * step
            w0  += counts[i] / total
            sum0 += counts[i] * mid / total
            w1   = 1.0 - w0
            if w0 == 0 or w1 == 0:
                continue
            mu0 = sum0 / w0
            mu1 = (total_mean - w0 * mu0) / w1
            between_var = w0 * w1 * (mu0 - mu1) ** 2
            if between_var > best_var:
                best_var    = between_var
                best_thresh = mn + (i + 1) * step

        return best_thresh

    def get_transitions(self) -> Tuple[Dict, Dict]:
       
        row_sums = {
            src: sum(dsts.values())
            for src, dsts in self.link_counts.items()
        }

        top_transitions: Dict = {}
        self_loop_rates: Dict = {}

        for src, dsts in self.link_counts.items():
            total_src = row_sums.get(src, 0)
            if total_src == 0:
                continue

            self_loop_cnt = dsts.get(src, 0)
            self_loop_rates[src] = round(self_loop_cnt / total_src, 4)

            candidates = []
            for dst, cnt in dsts.items():
                if dst == src:
                    continue
                prob = round(cnt / total_src, 4)
                candidates.append((dst, prob))

            if not candidates:
                top_transitions[src] = []
                continue

            # Sort by prob descending (replaces Borda + NPMI ranking)
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

            prob_vals = [c[1] for c in candidates]

            if len(prob_vals) <= 2:
                top_transitions[src] = candidates
                continue

            otsu_thresh = self.otsu_threshold(prob_vals)
            survivors = [c for c in candidates if c[1] >= otsu_thresh]

            if not survivors:
                survivors = [candidates[0]]

            top_transitions[src] = survivors

        return top_transitions, self_loop_rates


    def derive_tendency_thresholds(
        self, self_loop_rates: Dict
    ) -> Tuple[float, float]:
        non_zero = sorted([r for r in self_loop_rates.values() if r > 0])

        if len(non_zero) == 0:
            return 1.1, 1.1
        if len(non_zero) == 1:
            val = non_zero[0]
            return val, val
        if len(non_zero) == 2:
            return non_zero[0], non_zero[1]

        p33 = non_zero[len(non_zero) // 3]
        p66 = non_zero[(2 * len(non_zero)) // 3]
        print(f"  [tendency] p33={p33:.3f}  p66={p66:.3f}  "
              f"(from {len(non_zero)} actions)")
        return p33, p66

    def encode_hints(
        self,
        top_transitions: Dict,
        self_loop_rates: Dict,
    ) -> Dict:
     
        p33, p66 = self.derive_tendency_thresholds(self_loop_rates)

        hints: Dict = {}

        for src, candidates in top_transitions.items():
            action_key = self.index_to_action.get(src, str(src))
            entry: Dict = {}

            if candidates:
                entry["common_next"] = [
                    {
                        "action": self.index_to_action.get(dst, str(dst)),
                        "prob":   prob,
                    }
                    for dst, prob in candidates          # no npmi field
                ]
            else:
                entry["note"] = "terminal — sequence usually ends here"

            sl = self_loop_rates.get(src, 0.0)
            if sl >= p66:
                entry["repetition_tendency"] = "high"
            elif sl >= p33:
                entry["repetition_tendency"] = "moderate"

            hints[action_key] = entry

        return hints



def analyze_link(
    sequences: List[List[int]],
    actions: Dict[str, int],
    file_name: str,
) -> Dict:
   
    analyzer = LinkAnalyzer(actions)

    print("\n--- Step 1: build graph ---")
    analyzer.fit_sequences(sequences)
    print(f"  Actions seen: {len(analyzer.number_set)}")

    print("\n--- Step 2: sort by prob + local Otsu ---")
    top_transitions, self_loop_rates = analyzer.get_transitions()

    print("\n--- Step 3: encode LLM hints ---")
    hints = analyzer.encode_hints(top_transitions, self_loop_rates)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(hints, f, ensure_ascii=False, indent=4)

    print(f"\n--- Output ---")
    print(f"  Actions in hints : {len(hints)}")
    print(f"  Saved to         : {file_name}")

    return hints




def ATM(dataset: str, ori_env: str, actions: Dict[str, int]) -> Dict:
    print(f"\nGSS — dataset={dataset}  env={ori_env}")

    with open(f"IoT_data/{dataset}/{ori_env}/split_trn.pkl", "rb") as f:
        data = pickle.load(f)

    system_actions = {v for k, v in actions.items() if k.startswith("None:")}
    sequences = [
        [sublist[i] for i in range(3, len(sublist), 4) if sublist[i] not in system_actions]
        for sublist in data
    ]

    file_name = f'IoT_data/{dataset}/{ori_env}/action_transitions.json'
    hints = analyze_link(sequences, actions, file_name)

    print("\n--- LLM hints ---")
    print(json.dumps(hints, ensure_ascii=False, indent=2))
    return hints
