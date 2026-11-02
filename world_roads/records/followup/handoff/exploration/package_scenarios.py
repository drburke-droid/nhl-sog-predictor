import csv,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import era_field_adjustment as m
import opportunity_adjustment as o
import numpy as np

original=o.old  # opportunity_adjustment imports and patches m.newvalue; restore original scoring model
m.newvalue=original
res={}
for key,mode,strength in [('Original','raw',1),('Scoring era','era',0),('Scoring and field half','era_field',0.5),('Scoring and field full','era_field',1)]:
 _,_,r=m.get_results(mode,field_strength=strength)
 res[key]=r

def altered(perf,cid,sport,y,mode,field_strength=1.):
 if cid=='nba_ppg' and mode=='era':
  return float(perf)/o.chances[y] if y in o.chances else None
 return original(perf,cid,sport,y,mode,field_strength)
m.newvalue=altered
_,_,r=m.get_results('era',field_strength=0)
res['Scoring era with NBA shot opportunities']=r
m.newvalue=original

out=str(Path(__file__).resolve().parent / 'greatest_seasons_adjusted_scenarios.csv')
with open(out,'w',newline='') as f:
 w=csv.writer(f)
 w.writerow(['scenario','rank','athlete','season_end','sport','category','z_score','comparison_years'])
 for key,ds in res.items():
  for i,p in enumerate(ds,1):
   w.writerow([key,i,p['name'],p['year'],p['sport'],p['cat'],f'{p["z"]:.6f}',p['n']])

out2=str(Path(__file__).resolve().parent / 'greatest_seasons_adjusted_top20.csv')
with open(out2,'w',newline='') as f:
 w=csv.writer(f)
 w.writerow(['rank_original','athlete','year','category','original_z','scoring_era_rank','scoring_era_z','half_field_rank','half_field_z','full_field_rank','full_field_z','shot_opportunity_rank','shot_opportunity_z'])
 keys=['Original','Scoring era','Scoring and field half','Scoring and field full','Scoring era with NBA shot opportunities']
 look={k:{(v['sport'],m.norm(v['name']),v['year']):(i,v) for i,v in enumerate(res[k],1)} for k in keys}
 for i,v in enumerate(res['Original'][:20],1):
  id=(v['sport'],m.norm(v['name']),v['year'])
  fields=[]
  for key in keys:
   p=look[key].get(id)
   fields += [p[0] if p else '',f'{p[1]["z"]:.4f}' if p else '']
  w.writerow([i,v['name'],v['year'],v['cat'],fields[1],*fields[2:]])

print(out,out2)
for key in res:
 print('\n'+key)
 for i,p in enumerate(res[key][:10],1):print(f'{i:2} {p["name"]:24} {p["year"]} {p["z"]:6.3f}')
