"""Rebuild a joined comparison of original vs v2 scores using the original rank -> v2 stable-id crosswalk.
Run from anywhere: python repair_comparison.py. Requires followup/ from the user-supplied ZIP.
Does NOT change the original materials.
"""
import os,csv,sys,collections
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'followup')
sys.path.insert(0,os.path.join(ROOT,'analysis'))
import gsmodel as G
import context as X

def read(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

orig=read(os.path.join(ROOT,'data/original/season_lists.csv'))
v2=read(os.path.join(ROOT,'data/v2/season_lists_v2.csv'))
cats=G.Data('original').meta
byrank={(r['category'],int(r['season']),int(r['rank'])):r for r in orig}
byold=collections.defaultdict(set)
for r in v2:
    if not r['orig_rank'] or not r['athlete_id']:
        continue
    key=(r['category'],int(r['season']),int(r['orig_rank']))
    o=byrank.get(key)
    if o is None:continue
    sp=cats[r['category']]['sport']
    byold[(sp,G.norm(o['name_raw']),int(r['season']))].add(r['athlete_id'])

D0,D2=G.Data('original'),G.Data('v2')
gam={r['category']:float(r['gamma_pooled']) for r in read(os.path.join(ROOT,'outputs/gamma_estimates.csv'))}
ctx=X.make(D2,env='power',gamma=gam)
variants=[('original',D0,G.Spec(),None),('corrected',D2,G.Spec(ties='all'),None),('quadratic',D2,G.Spec(ties='all',trend='quad_safe'),None),('quadratic_environment',D2,G.Spec(ties='all',trend='quad_safe',env='power'),ctx)]
allvals={};display={};issues=[];collisions=[]
for key, data, spec, context in variants:
    recs=G.athlete_seasons(G.score(data,spec,ctx=context))
    for r in recs:
        k=(r['sport'],r['id'],r['year'])
        if key=='original':
            candidates=byold.get(k,set())
            if len(candidates)==1:
                k=(k[0],next(iter(candidates)),k[2])
            else:
                issues.append((r['sport'],r['name'],r['year'],r['cat'],r['z'],'ambiguous' if candidates else 'not mapped',','.join(sorted(candidates))))
                k=(k[0],'legacy:'+k[1],k[2])
        if key in allvals.get(k,{}):
            prev=allvals[k][key]
            collisions.append((r['sport'],r['year'],k[1],prev['name'],prev['z'],r['name'],r['z']))
            if prev['z'] >= r['z']:
                continue
        allvals.setdefault(k,{})[key]=r
        display.setdefault(k,(r['name'],r['cat']))

out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'comparison_join_fixed.csv')
fields=['sport','athlete','season','athlete_id','original_category','corrected_category','original_z','corrected_z','quadratic_z','quadratic_environment_z','delta_corrected_original','delta_quadratic_original','delta_environment_original']
with open(out,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for (sp,aid,yr),d in sorted(allvals.items(),key=lambda kv:(kv[0][0],str(kv[0][1]),kv[0][2])):
        nm=next((d[k]['name'] for k in ('quadratic_environment','quadratic','corrected','original') if k in d),display[(sp,aid,yr)][0])
        z={k:d[k]['z'] if k in d else None for k in ('original','corrected','quadratic','quadratic_environment')}
        w.writerow({'sport':sp,'athlete':nm,'season':yr,'athlete_id':aid,
                    'original_category':d.get('original',{}).get('cat',''),'corrected_category':d.get('corrected',{}).get('cat',''),
                    **{k+'_z':round(v,6) if v is not None else '' for k,v in z.items()},
                    'delta_corrected_original':round(z['corrected']-z['original'],6) if z['original'] is not None and z['corrected'] is not None else '',
                    'delta_quadratic_original':round(z['quadratic']-z['original'],6) if z['original'] is not None and z['quadratic'] is not None else '',
                    'delta_environment_original':round(z['quadratic_environment']-z['original'],6) if z['original'] is not None and z['quadratic_environment'] is not None else ''})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'comparison_join_unresolved.csv'),'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['sport','athlete','season','category','original_z','reason','candidate_ids']);w.writerows(issues)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'comparison_join_collisions.csv'),'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['sport','season','stable_id','original_name_a','z_a','original_name_b','z_b']);w.writerows(collisions)
print('rows:',len(allvals),'legacy rows:',len(issues),'matched original and corrected:',sum('original' in r and 'corrected' in r for r in allvals.values()),'original total:',sum('original' in r for r in allvals.values()),'corrected total:',sum('corrected' in r for r in allvals.values()))
print('legacy sample',issues[:12]);print('ID collisions:',collisions);print('unmatched original:',[(v['original']['name'],v['original']['year'],v['original']['cat']) for v in allvals.values() if 'original' in v and 'corrected' not in v]);print('unmatched corrected:',sum('corrected' in v and 'original' not in v for v in allvals.values()))
for nm,yr in [('Bob Beamon',1968),('Joe DiMaggio',1941),('Wilt Chamberlain',1962),('Wayne Gretzky',1986)]:
    d=[(v['original']['z'],v['corrected']['z'],v['quadratic']['z'],v['quadratic_environment']['z']) for v in allvals.values() if 'original' in v and v['original']['name']==nm and v['original']['year']==yr and all(k in v for k in ('corrected','quadratic','quadratic_environment'))]
    print(nm,yr, d)
