"""Generate deterministic input-quality flags, not verified corrections."""
from pathlib import Path
from collections import defaultdict
import csv
root=Path(__file__).resolve().parent.parent
inp=root/'original_project'/'inputs'/'season_lists.csv'
rows=list(csv.DictReader(inp.open(encoding='utf8',newline='')))
flags=[]
for r in rows:
    if int(r['season'])>2026:
        flags.append({'severity':'high','flag':'future_season','category':r['category'],'season':r['season'],'rank':r['rank'],'name':r['name_raw'],'detail':'Out of source snapshot boundary (November 2026); verify year against primary source'})
    if r['name_raw'].startswith(('+','-')) and any(x.isdigit() for x in r['name_raw'][:5]):
        flags.append({'severity':'medium','flag':'prefixed_name','category':r['category'],'season':r['season'],'rank':r['rank'],'name':r['name_raw'],'detail':'Possible stray wind/measurement prefix corrupting performer identity'})
seen=defaultdict(list)
for r in rows:
    seen[(r['category'],r['season'],r['name_raw'])].append(r)
for (cat,season,name),rs in seen.items():
    if len(rs)>1:
        flags.append({'severity':'medium','flag':'duplicate_name_same_category_season','category':cat,'season':season,'rank':';'.join(q['rank'] for q in rs),'name':name,'detail':f'{len(rs)} entries for same displayed performer; check distinct results versus duplicate entry'})
out=Path(__file__).resolve().parent/'data_quality_flags.csv'
with out.open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['severity','flag','category','season','rank','name','detail'])
    w.writeheader();w.writerows(flags)
print(f'Wrote {len(flags)} possible problems to {out}')
from collections import Counter
print(Counter(r['flag'] for r in flags))
