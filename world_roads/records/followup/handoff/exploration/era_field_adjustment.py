"""Era and field-size sensitivity using supplied top-ten season lists.
Source: Hockey-Reference league average goals per game through a GitHub tabulation;
Basketball-Reference NBA historical points per team per game through YonahAFeld's dashboard;
2025-26 from official-reference league average tables, 2026 incomplete season ignored.
"""
import os, csv, math, re, unicodedata
from collections import defaultdict
import numpy as np
P=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'original_project', 'inputs') + os.sep

# NBA team points per game by season-ending year. Source: Basketball Reference via YonahAFeld repository.
nbapoints='''
1947 67.8;1948 72.7;1949 80.0;1950 80.0;1951 84.1;1952 83.7;1953 82.7;1954 79.5;1955 93.1;1956 99.0;1957 99.6;1958 106.6;1959 108.2;1960 115.3;1961 118.1;1962 118.8;1963 115.3;1964 111.0;1965 110.6;1966 115.5;1967 117.4;1968 116.6;1969 112.3;1970 116.7;1971 112.4;1972 110.2;1973 107.6;1974 105.7;1975 102.6;1976 104.3;1977 106.5;1978 108.5;1979 110.3;1980 109.3;1981 108.1;1982 108.6;1983 108.5;1984 110.1;1985 110.8;1986 110.2;1987 109.9;1988 108.2;1989 109.2;1990 107.0;1991 106.3;1992 105.3;1993 105.3;1994 101.5;1995 101.4;1996 99.5;1997 96.9;1998 95.6;1999 91.6;2000 97.5;2001 94.8;2002 95.5;2003 95.1;2004 93.4;2005 97.2;2006 97.0;2007 98.7;2008 99.9;2009 100.0;2010 100.4;2011 99.6;2012 96.3;2013 98.1;2014 101.0;2015 100.0;2016 102.7;2017 105.6;2018 106.3;2019 111.2;2020 111.8;2021 112.1;2022 110.6;2023 114.7;2024 114.2;2025 113.8;2026 115.6'''
# NHL goals per team-game, including 2026 from Hockey-Reference league averages.
nhlgoals='''
1918 4.75;1919 4.13;1920 4.79;1921 4.23;1922 3.96;1923 3.26;1924 2.66;1925 2.50;1926 2.31;1927 2.00;1928 1.90;1929 1.46;1930 2.96;1931 2.40;1932 2.49;1933 2.28;1934 2.41;1935 2.52;1936 2.16;1937 2.46;1938 2.53;1939 2.53;1940 2.49;1941 2.68;1942 3.12;1943 3.61;1944 4.08;1945 3.68;1946 3.34;1947 3.16;1948 2.93;1949 2.72;1950 2.73;1951 2.71;1952 2.60;1953 2.40;1954 2.40;1955 2.52;1956 2.53;1957 2.69;1958 2.80;1959 2.90;1960 2.95;1961 3.00;1962 3.01;1963 2.97;1964 2.78;1965 2.88;1966 3.04;1967 2.98;1968 2.79;1969 2.98;1970 2.90;1971 3.12;1972 3.07;1973 3.28;1974 3.20;1975 3.43;1976 3.41;1977 3.32;1978 3.30;1979 3.50;1980 3.51;1981 3.84;1982 4.01;1983 3.86;1984 3.94;1985 3.89;1986 3.97;1987 3.67;1988 3.71;1989 3.74;1990 3.68;1991 3.46;1992 3.48;1993 3.63;1994 3.24;1995 2.99;1996 3.14;1997 2.92;1998 2.64;1999 2.63;2000 2.75;2001 2.76;2002 2.62;2003 2.65;2004 2.57;2006 3.03;2007 2.88;2008 2.72;2009 2.85;2010 2.77;2011 2.73;2012 2.66;2013 2.65;2014 2.67;2015 2.66;2016 2.67;2017 2.73;2018 2.93;2019 2.98;2020 2.98;2021 2.90;2022 3.11;2023 3.14;2024 3.08;2025 3.01;2026 3.08'''
def parse(s): return {int(y):float(v) for z in s.strip().split(';') for y,v in [z.split()]}
ppg=parse(nbapoints);gpg=parse(nhlgoals)

def nba_teams(y):
 if y==1947:return 11
 if y==1948:return 8
 if y==1949:return 12
 if y==1950:return 17
 if y==1951:return 11
 if y in (1952,1953):return 10
 if y in (1954,1955):return 9
 if 1956<=y<=1961:return 8
 if 1962<=y<=1966:return 9
 if y==1967:return 10
 if y==1968:return 12
 if y in (1969,1970):return 14
 if 1971<=y<=1974:return 17
 if 1975<=y<=1976:return 18
 if 1977<=y<=1980:return 22
 if 1981<=y<=1988:return 23
 if y==1989:return 25
 if 1990<=y<=1995:return 27
 if 1996<=y<=2004:return 29
 return 30

def nhl_teams(y):
 if y<=1924:return 4
 if y==1925:return 6
 if y==1926:return 7
 if y<=1931:return 10
 if y<=1934:return 9
 if y<=1938:return 8
 if y<=1942:return 7
 if y<=1967:return 6
 if y<=1970:return 12
 if y<=1972:return 14
 if y<=1974:return 16
 if y<=1978:return 18
 if y==1979:return 17
 if y<=1991:return 21
 if y==1992:return 22
 if y==1993:return 24
 if y<=1998:return 26
 if y==1999:return 27
 if y==2000:return 28
 if y<=2017:return 30
 if y<=2021:return 31
 return 32

def norm(name):
 s=unicodedata.normalize('NFKD',name);return ' '.join(re.sub(r'[^a-z ]','', ''.join(c for c in s if not unicodedata.combining(c)).lower()).split())
C={r['category']:r for r in csv.DictReader(open(P+'categories.csv'))}
S=defaultdict(lambda:defaultdict(list))
for r in csv.DictReader(open(P+'season_lists.csv')):S[r['category']][int(r['season'])].append(r)

def env(sport,y):return (ppg if sport=='NBA' else gpg).get(y) if sport in ('NBA','NHL') else None
def teams(sport,y):return nba_teams(y) if sport=='NBA' else nhl_teams(y) if sport=='NHL' else None

# Upper tail scale beta: slope of rank's performance versus minus log(rank - .5)
# For each season, omit the champion to avoid self-influence; use ranks 2-10.
# OLS rank-tail slopes fitted per year, then median-smoothed over +/-7 calendar years.
# This is NOT a Theil-Sen estimate and should NOT be called robust regression.
beta0={}
for sport in ['NHL','NBA']:
 for cat, c in C.items():
  if c['sport']!=sport:continue
  for y, top10 in S[cat].items():
   if len(top10)<6 or env(sport,y) is None:continue
   x=np.array([float(t['performance'])/env(sport,y) for t in top10[1:]],dtype=float)
   a=np.array([-math.log(i-0.5) for i in range(2,len(x)+2)])
   # Ordinary least squares slope of adjusted top ranks; beta units in 'league scoring equivalents'
   beta=float(np.polyfit(a,x,1)[0]);beta0[cat,y]=max(0.0,beta)
# pool to estimate beta by local adjacent years, reducing overfit to one weird top ten
betas={}
for (cat,y) in beta0:
 near=[b for (cc,yy),b in beta0.items() if cc==cat and abs(yy-y)<=7]
 betas[cat,y]=float(np.median(near)) if len(near)>=4 else float(np.median([v for (cc,yy),v in beta0.items() if cc==cat]))

def newvalue(perf,cid,sport,y,mode,field_strength=1.):
 v=float(perf)
 if sport not in ('NBA','NHL') or mode=='raw':return v
 # Rebounds, assists and threes are not proportional to total NBA scoring.
 # Apply league-size correction to those categories, but no scoring-environment correction.
 non_scoring = sport=='NBA' and cid!='nba_ppg'
 e=env(sport,y)
 if e is None:return None
 v=v/e if not non_scoring else v  # raw units for NBA non-scoring categories
 if mode=='field_only' and (cid,y) not in betas:return float(perf)
 if mode in ('era_field','field_only'):
  # normalize to reference 30/32-team league. Theoretical Gumbel max offset beta*log(n_ref/n).
  ref=30 if sport=='NBA' else 32
  beta=betas.get((cid,y))
  if beta is not None:
   if mode=='field_only' or non_scoring:
    # In raw units, raw-tail scale must be beta*era environment
    v=float(perf)+field_strength*beta*e*math.log(ref/teams(sport,y))
   else:
    v+=field_strength*beta*math.log(ref/teams(sport,y))
 if mode=='era':pass
 return v

def measure(pts,y,x,shift):
 yr=np.array([p[0] for p in pts],float);v=np.array([p[1] for p in pts],float)
 d=np.isin(yr,list(shift)).astype(float);has_shift=bool(shift and d.sum()>0 and len(yr)-d.sum()>=8)
 X=np.column_stack([yr-y,np.ones_like(yr)]+([d] if has_shift else []))
 coef=np.linalg.lstsq(X,v,rcond=None)[0];res=v-X@coef
 sd=math.sqrt(float(res@res)/(len(v)-len(coef))) if len(v)>len(coef) else np.nan
 base=coef[1]+(coef[2] if has_shift and y in shift else 0)
 return (x-base)/sd if sd>0 else float('nan')

def get_results(mode='raw',field_strength=1.,window=15):
 rows=[]
 for cid,c in C.items():
  sport=c['sport'];short={int(t) for t in c['short_seasons_left_out'].split()};shift={int(t) for t in c['level_shift_seasons'].split()}
  scaled=c['is_total_scaled_by_schedule']=='True';cut=float(c['cutoff_performance']) if c['cutoff_performance'] else None
  first,start=int(c['first_season']),int(c['performances_count_from'])
  for y in sorted(S[cid]):
   if y in short or y<start:continue
   lead=S[cid][y][0];nm=norm(lead['name_raw']);gy=float(lead['games_per_team']) if scaled else 1.0
   x=newvalue(lead['performance'],cid,sport,y,mode,field_strength)
   if x is None:continue
   pts=[];real=0
   for s in range(y-window,y+window+1):
    if s not in S[cid] or s in short:continue
    o=next((e for e in S[cid][s] if norm(e['name_raw'])!=nm),None)
    if o is None:
     if cut is not None and s>=first:pts.append((s,cut))
     continue
    v=newvalue(o['performance'],cid,sport,s,mode,field_strength)
    if v is None:continue
    if scaled:v=v*gy/float(o['games_per_team'])
    pts.append((s,v));real+=1
   if real<8:continue
   z=measure(pts,y,x,shift)
   if np.isfinite(z):rows.append(dict(cat=cid,sport=sport,name=lead['name_display'],year=y,z=z,n=len(pts),raw=lead['raw_value']))
 best={}
 for r in rows:
  k=(r['sport'],norm(r['name']),r['year'])
  if k not in best or best[k]['z']<r['z']:best[k]=r
 ranked=sorted(best.values(),key=lambda r:-r['z']);distinct=[];seen=set()
 for r in ranked:
  k=(r['sport'],norm(r['name']))
  if k not in seen: distinct.append(r);seen.add(k)
 return rows,ranked,distinct

if __name__=='__main__':
 print('scoring 1986 NHL',gpg[1986],'2026',gpg[2026],'Wilt PPG 1962',ppg[1962],'2026',ppg[2026])
 print('NHL beta 1986 assists', beta0.get(('nhl_assists',1986)),betas.get(('nhl_assists',1986)))
 from pprint import pprint
 res={k:get_results(k) for k in ('raw','era','field_only','era_field')}
 comp={}
 for k,(rows,ranked,distinct) in res.items():
  print('\n',k,'season/cat scores',len(rows),'ranked',len(ranked))
  for i,r in enumerate(distinct[:20],1): print(f'{i:2d} {r["name"][:25]:25} {r["year"]} {r["cat"][:17]:17} z={r["z"]:7.3f}')
  for nm in ['Wilt Chamberlain','Wayne Gretzky','Dennis Rodman','Joe DiMaggio','Bob Beamon','Pete Rose']:
   m=next((r for r in distinct if r['name']==nm),None)
   if m:comp[k,nm]=(m['z'],distinct.index(m)+1,m['cat'],m['year'])
 print('\nSUMMARY\n')
 for nm in ['Wilt Chamberlain','Wayne Gretzky','Dennis Rodman','Joe DiMaggio','Bob Beamon','Pete Rose']:
  print(nm,[(k,comp.get((k,nm))) for k in res])
 out=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'era_field_sensitivity.csv')
 with open(out,'w',newline='') as f:
  w=csv.writer(f);w.writerow(['model','rank','athlete','year','category','sport','z','n_window'])
  for k,(_,_,distinct) in res.items():
   for rank,r in enumerate(distinct,1):w.writerow([k,rank,r['name'],r['year'],r['cat'],r['sport'],round(r['z'],7),r['n']])
 print('saved',out)
