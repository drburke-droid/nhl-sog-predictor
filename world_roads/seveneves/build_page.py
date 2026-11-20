"""Fill page_template.html with results/results.json -> results/seveneves.html (the findings page)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, 'results', 'results.json')))
for v, sc in data['scenarios'].items():         # the page needs only a few summary fields
    if v != '150':                               # charts show the 150 m/s run only
        sc.pop('series', None); sc.pop('series_no_breakup_debris', None); sc.pop('dial', None)
    s = sc['summary']
    sc['summary'] = dict(rain=s['rain'], climate=s['climate'], config=dict(debris0=s['config']['debris0']),
                         budget=s['budget_frac_of_moon'])
li = json.load(open(os.path.join(HERE, 'results', 'lunar_impact.json')))
keep = ['20kms_rock_average', '20kms_rock_trailing', '20kms_regolith_average', '20kms_regolith_trailing',
        '12kms_rock_average', '50kms_rock_average']
data['lunar_impact'] = dict(rungs=li['rungs'], cases={k: dict(thresholds=li['cases'][k]['thresholds'],
                            frac_to_earth=li['cases'][k]['frac_to_earth_30y'],
                            curve=[[r['D_km'], round(r['warming_C'], 2), r['E_total']] for r in li['cases'][k]['rows'][::2]])
                            for k in keep})
lad = json.load(open(os.path.join(HERE, 'results', 'impact_ladder.json')))
data['ladder'] = [dict(D=r['D_m'], crater=r['crater_km'], earth=r['to_earth_kg'], storm=r['storm_factor'],
                       chely=r['chelyabinsks'], chely_hi=r['chelyabinsks_range'][1], tung=r['tunguskas'],
                       tung_hi=r['tunguskas_range'][1], warm=r['warming_C'], ocean=r['ocean'],
                       moon=r['moon_interval_yr'], earthint=r['earth_interval_yr'])
                  for r in lad['rows'] if r['material'] == 'rock']
html = open(os.path.join(HERE, 'page_template.html')).read().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
out = os.path.join(HERE, 'results', 'seveneves.html')
open(out, 'w').write(html)
print('wrote', out, f'{len(html) / 1024:.0f} KB')
