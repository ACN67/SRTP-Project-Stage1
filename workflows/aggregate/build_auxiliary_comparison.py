#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/auxiliary/full_eval/comparison.csv'

def read_rows(path: Path) -> list[dict[str,str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row.setdefault('evidence_status','aggregate_only')
        row['evidence_status']='aggregate_only'
    return rows

def csv_text(rows: list[dict[str,str]]) -> str:
    import io
    if not rows:
        return ''
    fields=list(rows[0].keys())
    if 'evidence_status' not in fields:
        fields.append('evidence_status')
    buf=io.StringIO()
    writer=csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='Validate auxiliary full-evaluation aggregate table.')
    parser.add_argument('--write', action='store_true', help='Validate and keep the aggregate table when raw evidence is unavailable.')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'results/auxiliary/full_eval/comparison.csv')
    args=parser.parse_args(argv)
    path=args.output if args.output.is_absolute() else ROOT/args.output
    source = path if path.exists() and path.resolve() == SOURCE.resolve() else SOURCE
    rows=read_rows(source)
    text=csv_text(rows)
    ok=bool(rows) and {'method','model','benchmark','metric','value','evidence_status'} <= set(rows[0])
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    elif args.check:
        ok = ok and path.exists() and path.read_text(encoding='utf-8') == text
    print(json.dumps({'evidence_status':'aggregate_only','row_count':len(rows),'ok':ok}, indent=2))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main(sys.argv[1:]))
