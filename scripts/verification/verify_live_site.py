import urllib.request
import json
import re

base_url = "https://rajuriprathik-a11y.github.io/next-move-ai"
headers = {"User-Agent": "NextMoveAI-Verifier"}

print(f"Testing live site at: {base_url}/")
req = urllib.request.Request(f"{base_url}/", headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8')
        print(f"Index HTML: HTTP {resp.status}, length: {len(html)}")
        sections = re.findall(r'<section\s+id=[\'"]([^\'"]+)[\'"]', html)
        print("Sections found:", sections)
        print("Hero present:", 'Next Move AI' in html)
        print("Validation section present:", 'Validation Strategy & Data Splitting' in html)
        print("Authoritative metrics present:", '0.54747' in html and '0.54972' in html)
        print("Section 13 Simulator present:", 'id="simulator"' in html)
except Exception as e:
    print(f"Error fetching index: {e}")

# Check all 6 JSON endpoints on the live site
json_files = [
    'curated_scenarios.json',
    'oof_threshold_metrics.json',
    'representative_questions.json',
    'results.json',
    'training_priors.json',
    'xgboost_trees.json'
]

for jf in json_files:
    jurl = f"{base_url}/data/{jf}"
    try:
        jreq = urllib.request.Request(jurl, headers=headers)
        with urllib.request.urlopen(jreq) as jresp:
            content = jresp.read()
            data = json.loads(content.decode('utf-8'))
            print(f"Live JSON /data/{jf}: HTTP {jresp.status}, bytes: {len(content)}, parsed: {type(data).__name__}")
    except Exception as e:
        print(f"Live JSON /data/{jf} ERROR: {e}")
