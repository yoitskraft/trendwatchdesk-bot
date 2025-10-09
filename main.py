#!/usr/bin/env python3
# -*- coding: utf-8 -*-

name: TrendWatchDesk Generate (Charts + Posters)

on:
  schedule:
    - cron: "10 7 * * 1,3,5"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Show Python & pip
        run: |
          python --version
          pip --version

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install --upgrade yfinance pillow pandas numpy requests pytz urllib3

      - name: Ensure output folders
        run: |
          mkdir -p output output/charts output/posters

      # (Optional) Auto-fill logos/badges before running main
      - name: Fetch missing logos
        run: |
          python tools/download_logos.py || true

      - name: Run charts + posters
        env:
          PYTHONUNBUFFERED: "1"
        run: |
          set -x
          python -u main.py --both 2>&1 | tee run.log

      - name: Print run.log to console
        if: always()
        run: |
          echo "===== BEGIN run.log ====="
          (test -f run.log && cat run.log) || echo "run.log missing"
          echo "===== END run.log ====="

      - name: Show output tree
        if: always()
        run: |
          echo "== output =="
          ls -la output || true
          echo "== charts =="
          ls -la output/charts || true
          echo "== posters =="
          ls -la output/posters || true
          echo "== captions =="
          ls -la output/caption_* 2>/dev/null || true

      # Require at least charts OR posters (relax as needed)
      - name: Require images
        if: always()
        run: |
          set -e
          CHARTS=$(find output/charts -type f -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
          POSTERS=$(find output/posters -type f -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
          echo "Charts:  $CHARTS"
          echo "Posters: $POSTERS"
          if [ "$CHARTS" = "0" ] && [ "$POSTERS" = "0" ]; then
            echo "::error::No images generated. Check run.log above for errors."
            exit 1
          fi

      - name: Publish to 'charts' branch (single-commit)
        if: success()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO: ${{ github.repository }}
        run: |
          set -e
          rm -rf publish && mkdir publish
          cp -r output/* publish/
          cd publish
          git init
          git checkout -b charts
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git"
          git add -A
          git commit -m "Publish outputs $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
          git push -f origin charts

      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: twd-run-${{ github.run_id }}
          path: |
            run.log
            output/**
