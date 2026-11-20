"""Fill page_template.html with results/results.json -> results/seveneves.html (the findings page)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, 'results', 'results.json')))
for sc in data['scenarios'].values():           # the page needs only a few summary fields
    s = sc['summary']
    sc['summary'] = dict(rain=s['rain'], climate=s['climate'], config=dict(debris0=s['config']['debris0']),
                         budget=s['budget_frac_of_moon'])
html = open(os.path.join(HERE, 'page_template.html')).read().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
out = os.path.join(HERE, 'results', 'seveneves.html')
open(out, 'w').write(html)
print('wrote', out, f'{len(html) / 1024:.0f} KB')
