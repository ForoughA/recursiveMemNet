import sys
import os
import json
folder = sys.argv[1]
head = sys.argv[2]
stat_list = []
sizes = [277] * 14 + [281]
total = 4159
for file in os.listdir(folder):
    if not file.startswith(head) or not file.endswith(".json") or "agg" in file:
        continue
    path = os.path.join(folder, file)
    with open(path) as f:
        stats = json.loads(f.read())
        stat_list.append(stats)
assert len(stat_list) == 15
epoch = stat_list[0]["epoch"]
set_ = stat_list[0]["set"]
agg = {"type":"eval", "epoch":epoch, "set": set_, "stats":{}}
for depth in set(sum((list(stat["stats"].keys()) for stat in stat_list),[])):
    agg["stats"][depth] = {}
    for statistic in stat_list[0]["stats"]["all"]:
        numerator = 0
        denominator = 0
        for stat in stat_list:
            if depth in stat["stats"]:
                numerator += stat["stats"][depth]["count"] * stat["stats"][depth][statistic]
                denominator += stat["stats"][depth]["count"]
        agg["stats"][depth][statistic] = numerator / denominator if statistic != "count" else denominator
with open(os.path.join(folder, head+"agg.json"), "w") as f:
    json.dump(agg, f, indent=4, sort_keys=True)
