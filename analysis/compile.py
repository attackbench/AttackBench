import argparse
import json
import pathlib
from collections import defaultdict

from read import read_results

parser = argparse.ArgumentParser('Compile results from several attacks')

parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')

args = parser.parse_args()

# check that result directory exists
result_path = pathlib.Path(args.dir)
assert result_path.exists()

# find info files corresponding to finished experiments
info_files = result_path.glob('**/info.json')

results = defaultdict(list)

# traverse result files
for info_file in info_files:
    scenario, hash_distances = read_results(info_file=info_file)
    results[scenario].append((info_file.parent, hash_distances))

# compile and save best distances
for scenario in results.keys():
    best_distances = {}

    for attack_dir, hash_distances in results[scenario]:
        attack_path = attack_dir.relative_to(result_path)
        for hash, distance in hash_distances.items():
            best_distance = best_distances.get(hash, (None, float('inf')))[1]
            if distance < best_distance:  # TODO: fix ambiguous case where two attacks have the same best distance
                best_distances[hash] = (attack_path.as_posix(), distance)

    with open(result_path / f'{"-".join(scenario)}.json', 'w') as f:
        json.dump(best_distances, f)
