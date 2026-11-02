"""Reproduction checks against preliminary scores. Run from any working directory."""
import csv, math
import era_field_adjustment as m

def find(rows, name):
    return next((v for v in rows if v['name']==name), None)

expected = {
 'raw': {'Wilt Chamberlain': 7.731371, 'Wayne Gretzky': 7.499599, 'Joe DiMaggio': 7.197192},
 'era': {'Wilt Chamberlain': 4.4778, 'Wayne Gretzky': 5.1947, 'Joe DiMaggio': 7.197192},
 'era_field': {'Wilt Chamberlain': 6.3275, 'Wayne Gretzky': 5.1693, 'Joe DiMaggio': 7.197192},
}
for mode, checks in expected.items():
    _, _, d = m.get_results(mode, field_strength=1.,window=15)
    for name, expected_z in checks.items():
        row = find(d, name)
        assert row, f'Missing {name} in {mode}'
        assert abs(row['z']-expected_z)<0.002, f'{mode} {name}: got {row["z"]}, expected {expected_z}'
        print(f'PASS {mode:10} {name:22} z={row["z"]:.6f}; rank={d.index(row)+1}')
print('All preliminary numerical cross-checks passed.')
