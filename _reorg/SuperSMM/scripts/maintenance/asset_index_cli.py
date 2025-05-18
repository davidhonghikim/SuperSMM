#!/usr/bin/env python3
"""
CLI utility for searching and displaying SuperSMM PROJECT_ASSET_INDEX.
Usage:
  python3 asset_index_cli.py [search_term]
  python3 asset_index_cli.py --all
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PROJECT_ROOT, 'PROJECT_ASSET_INDEX.json')

if not os.path.isfile(INDEX_PATH):
    print("[ERROR] PROJECT_ASSET_INDEX.json not found. Run update_asset_index.py first.")
    sys.exit(1)

with open(INDEX_PATH, 'r') as f:
    assets = json.load(f)

def print_asset(asset):
    print(f"- Path: {asset['path']}")
    print(f"  Type: {asset['type']}")
    print(f"  Purpose: {asset['purpose']}")
    print(f"  Related: {', '.join(asset.get('related', [])) or '-'}\n")

def search(term):
    found = False
    for asset in assets:
        if term.lower() in asset['path'].lower() or term.lower() in asset['purpose'].lower():
            print_asset(asset)
            found = True
    if not found:
        print(f"[No assets found matching '{term}']")

def show_all():
    for asset in assets:
        print_asset(asset)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] != '--all':
        search(sys.argv[1])
    else:
        show_all()
