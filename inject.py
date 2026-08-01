import glob
import re
import os

injection_code = """
# -- INJECT TELEMETRY FOR FRONTEND --
if 'global_telemetry' in globals():
    local_status = locals().get('is_fully_correct', locals().get('is_ready', False))
    local_msgs = locals().get('instructions', locals().get('checklist', []))
    fail_msgs = [m for m in local_msgs if "[FAIL]" in m or "[X]" in m]
    if local_status:
        global_telemetry['message'] = "Perfect alignment. Keep holding."
        global_telemetry['accuracy'] = 95
        global_telemetry['status'] = "good"
    elif fail_msgs:
        global_telemetry['message'] = fail_msgs[0].replace("[FAIL] ", "Warning: ").replace("[X] ", "Warning: ")
        global_telemetry['accuracy'] = 45
        global_telemetry['status'] = "bad"
    else:
        global_telemetry['message'] = "Analyzing..."
        global_telemetry['accuracy'] = 10
        global_telemetry['status'] = "calibrating"
import time
time.sleep(0.01) # Yield GIL
"""

count = 0
import glob
files_to_inject = []
for directory in ["chest", "elbow", "foot", "hand", "knee", "lower_back", "spine"]:
    files_to_inject.extend(glob.glob(f"backend/{directory}/**/*.py", recursive=True))

for filepath in files_to_inject:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "INJECT TELEMETRY FOR FRONTEND" in content:
        continue
        
    # Find the line that starts with spaces then frame_placeholder.image(
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.lstrip().startswith("frame_placeholder.image("):
            indent = line[:len(line) - len(line.lstrip())]
            for inj_line in injection_code.strip().split('\n'):
                new_lines.append(indent + inj_line)
            new_lines.append(line)
        else:
            new_lines.append(line)
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.write('\n'.join(new_lines))
        print(f"Injected telemetry into {filepath}")
        count += 1

print(f"Total files injected: {count}")
