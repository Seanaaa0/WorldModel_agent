import os
import re

LOG_DIR = "outputs/fast_predictive_legacy_15x15_seeded_debug"

success_count = 0
fail_count = 0
steps_list = []

for file in os.listdir(LOG_DIR):

    if not file.endswith(".txt"):
        continue

    path = os.path.join(LOG_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    result_match = re.search(r"RESULT:\s*(SUCCESS|FAIL)", text)
    steps_match = re.search(r"STEPS:\s*(\d+)", text)

    if not result_match:
        continue

    result = result_match.group(1)

    if result == "SUCCESS":
        success_count += 1

        if steps_match:
            steps = int(steps_match.group(1))
            steps_list.append(steps)

    else:
        fail_count += 1


total = success_count + fail_count

print("=================================")
print("Experiment Summary")
print("=================================")
print("DIR: ", LOG_DIR)
print("Total episodes:", total)
print("Success:", success_count)
print("Fail:", fail_count)

if total > 0:
    print("Success rate:", success_count / total)

if steps_list:
    avg_steps = sum(steps_list) / len(steps_list)
    print("Average steps (success only):", avg_steps)
    print("Min steps:", min(steps_list))
    print("Max steps:", max(steps_list))
