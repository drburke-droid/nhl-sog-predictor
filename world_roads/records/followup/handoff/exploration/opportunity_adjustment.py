"""NBA per-game shot opportunity normalization (FGA + .44 FTA) sensitivity.
This is a proxy rather than true possessions, to avoid pretending pre-1974 turnover/offensive rebound stats are known.
Other sports use previous scoring-era normalization. No field size adjustment in this test.
"""
import era_field_adjustment as m
from pathlib import Path

# Season end-year, FGA, FTA from Basketball Reference NBA league per-game averages,
# via YonahAFeld/NBA-League-Dashboard data/Avg league stats per game.csv, 1947-2025.
raw='''
1947 92.9 24.8;1948 96 27;1949 88.7 31.3;1950 83.1 33;1951 83.6 33.4;1952 80.8 33.2;1953 77.1 35.9;1954 75.4 33;1955 86.4 35.9;1956 91.4 38;1957 94.6 36.9;1958 101.9 38.3;1959 102.3 36.3;1960 108.7 35.8;1961 109.4 37.4;1962 107.7 37.1;1963 101.2 35.9;1964 99.1 35;1965 99.8 35.6;1966 102.4 37;1967 103.1 36.2;1968 100.8 37.1;1969 99 35;1970 99.4 33.7;1971 98 32.7;1972 95.5 31.2;1973 96.9 25.3;1974 93.9 25.4;1975 91.1 25.2;1976 91.7 26.9;1977 92 27.7;1978 92.9 28.4;1979 91.7 28.3;1980 90.6 27.8;1981 88.4 28.9;1982 88.2 28.6;1983 89.7 28.3;1984 88.4 29.7;1985 89.1 29.4;1986 88.6 30.3;1987 88.8 30.5;1988 87.7 29.1;1989 89 28.8;1990 87.2 28.5;1991 87.2 27.9;1992 87.3 26.7;1993 86 27.7;1994 84.4 26.6;1995 81.5 27.1;1996 80.2 26.4;1997 79.3 25.3;1998 79.7 26.3;1999 78.2 25.8;2000 82.1 25.3;2001 80.6 24.9;2002 81.3 23.8;2003 80.8 24.4;2004 79.8 24.2;2005 80.3 26.1;2006 79 26.3;2007 79.7 26.1;2008 81.5 24.9;2009 80.9 24.7;2010 81.7 24.5;2011 81.2 24.4;2012 81.4 22.5;2013 82 22.2;2014 83 23.6;2015 83.6 22.8;2016 84.6 23.4;2017 85.4 23.1;2018 86.1 21.7;2019 89.2 23.1;2020 88.8 23.1;2021 88.4 21.8;2022 88.1 21.9;2023 88.3 23.5;2024 88.9 21.7;2025 89.2 21.8;2026 89.1 23.5'''
chances={int(y):float(fga)+.44*float(fta) for row in raw.strip().split(';') for y,fga,fta in [row.strip().split()]}
old=m.newvalue

def newvalue(perf,cid,sport,y,mode,field_strength=1.):
 if cid=='nba_ppg' and mode=='era':
  if y not in chances:return None
  return float(perf)/chances[y]
 return old(perf,cid,sport,y,mode,field_strength)
m.newvalue=newvalue

if __name__=='__main__':
 print('Wilt shot opportunity factor',chances[1962],'2026',chances[2026],'adjusted into 2026',50.4*chances[2026]/chances[1962])
 print('1962 pace reported 126.2 2026 pace 99.4; modern pace-equivalent PPG',50.4*99.4/126.2)
 for w in [7,10,15,20]:
  rows,ranked,distinct=m.get_results('era',window=w)
  for name in ['Wilt Chamberlain','Wayne Gretzky']:
   r=next(r for r in distinct if r['name']==name)
   print(w,name,'z',round(r['z'],3),'rank',distinct.index(r)+1,'cat',r['cat'])
 print('top 10 ±15 with Wilt shot opportunity model')
 _,_,distinct=m.get_results('era',window=15)
 for i,r in enumerate(distinct[:10],1):print(i,r['name'],r['year'],r['cat'],round(r['z'],3))

# Combined scenario, with the same rank-tail field-size mechanism
# calibrated in shot-opportunity rather than team-points units for NBA PPG.
if __name__=='__main__':
 import numpy as np,math
 beta0={}
 for y, top10 in m.S['nba_ppg'].items():
  if len(top10)<6 or y not in chances:continue
  xx=np.array([float(t['performance'])/chances[y] for t in top10[1:]])
  a=np.array([-math.log(i-0.5) for i in range(2,len(xx)+2)])
  beta0[y]=max(0.,float(np.polyfit(a,xx,1)[0]))
 betas={y:float(np.median([b for yy,b in beta0.items() if abs(yy-y)<=7])) for y in beta0}
 def combined(perf,cid,sport,y,mode,field_strength=1.):
  if cid=='nba_ppg' and mode in ('era','era_field'):
   v=float(perf)/chances[y] if y in chances else None
   if v is None:return None
   if mode=='era_field' and y in betas:
    v+=field_strength*betas[y]*math.log(30/m.nba_teams(y))
   return v
  return old(perf,cid,sport,y,mode,field_strength)
 m.newvalue=combined
 for s in [0,.25,.5,1]:
  _,_,d=m.get_results('era_field',field_strength=s)
  print('OPPORTUNITY + FIELD STRENGTH',s)
  for nm in ['Joe DiMaggio','Wilt Chamberlain','Wayne Gretzky','Babe Dye','Stephen Curry']:
   r=next((r for r in d if r['name']==nm),None)
   if r:print(nm,d.index(r)+1,round(r['z'],3))
  print('Top 10', [(r['name'],r['year'],round(r['z'],3)) for r in d[:10]])
