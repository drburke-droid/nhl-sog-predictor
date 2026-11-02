import era_field_adjustment as m
from pathlib import Path
p=Path(__file__).resolve().parent / 'era_field_robustness.txt'
with p.open('w') as f:
 def pr(*x): print(*x,file=f)
 for st in [0,0.25,0.5,1.0,1.5,2.0]:
  _,ranked,distinct=m.get_results('era_field',field_strength=st)
  pr('STRENGTH',st)
  for nm in ['Wilt Chamberlain','Wayne Gretzky','Joe DiMaggio','Babe Dye','Stephen Curry','Dennis Rodman']:
   r=next((r for r in distinct if r['name']==nm),None)
   if r:pr(nm, f'z={r["z"]:.3f}', f'rank={distinct.index(r)+1}',f'cat={r["cat"]}')
 for window in [7,10,15,20]:
  _,_,distinct=m.get_results('era_field',field_strength=1.,window=window)
  pr('\nWINDOW',window)
  for nm in ['Wilt Chamberlain','Wayne Gretzky','Joe DiMaggio']:
   r=next((r for r in distinct if r['name']==nm),None)
   if r:pr(nm, f'z={r["z"]:.3f}', f'rank={distinct.index(r)+1}')
print(p)
