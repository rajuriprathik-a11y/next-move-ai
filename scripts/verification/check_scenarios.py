import json

with open('website/data/curated_scenarios.json', 'r', encoding='utf-8') as f:
    scenarios = json.load(f)

print(f"Total curated scenarios: {len(scenarios)}")
for i, s in enumerate(scenarios):
    print(f"{i+1}. {s.get('name')} | overall: {s.get('overall_correct')}/{s.get('overall_interactions')} | same_skill: {s.get('same_skill_correct')}/{s.get('same_skill_interactions')} | other_skill: {s.get('other_skill_correct')}/{s.get('other_skill_interactions')}")
