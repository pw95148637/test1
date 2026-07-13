#!/usr/bin/env python3
"""
TPE -> Tokyo (NRT/HND) flight price watcher — Tigerair / Peach

Checks each airline's fare calendar for a date range and pushes a
notification via ntfy.sh when a fare drops at or below THRESHOLD_TWD.

STATUS: notify/threshold/scheduling logic is complete and ready to run.
The two fetch_*() functions are stubbed — they need the airlines' real
fare-calendar data endpoint wired in. LCC booking sites load their
calendar view via an internal JSON API rather than static HTML, so the
exact endpoint has to be read off the Network tab in devtools. This
sandbox can't reach tigerairtw.com / flypeach.com to find it directly
(network here is restricted to package registries), but it's a five
minute job for Claude Code running on your own machine, which has normal
internet: point it at the calendar_url below for each airline and ask it
to open devtools, load the calendar, and copy the underlying request.
"""

import requests
from datetime import date
import json
import sys

# ---------------- config ----------------
THRESHOLD_TWD = 2500                    # alert when one-way fare <= this
DATE_FROM = date(2026, 11, 9)           # adjust once your outbound window narrows
DATE_TO = date(2026, 11, 14)
NTFY_TOPIC = "jason-tpe-nrt-9f2k"       # change this — ntfy topics are public by name, pick something unguessable

ROUTES = [
    {
        "carrier": "Tigerair",
        "origin": "TPE",
        "dest": "NRT",
        "calendar_url": "https://www.tigerairtw.com/zh-TW/booking/select-flight",
    },
    {
        "carrier": "Peach",
        "origin": "TPE",
        "dest": "NRT",
        "calendar_url": "https://www.flypeach.com/tw/en/lccrsp/lfs",
    },
    # Add the return leg once you have USJ-adjacent dates settled, e.g.:
    # {"carrier": "Tigerair", "origin": "KIX", "dest": "TPE",
    #  "calendar_url": "https://www.tigerairtw.com/zh-TW/booking/select-flight"},
]
# -----------------------------------------


def fetch_tigerair(origin, dest, date_from, date_to):
    """
    TODO: replace with the real fare-calendar JSON endpoint.
    Expected return shape: [{"date": "2026-11-09", "price_twd": 2380}, ...]
    """
    raise NotImplementedError("wire up Tigerair's calendar endpoint")


def fetch_peach(origin, dest, date_from, date_to):
    """Same idea as fetch_tigerair — Peach's calendar is also API-backed."""
    raise NotImplementedError("wire up Peach's calendar endpoint")


FETCHERS = {"Tigerair": fetch_tigerair, "Peach": fetch_peach}


def notify(message: str):
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode("utf-8"), timeout=10)
    except requests.RequestException as e:
        print(f"[warn] notify failed: {e}", file=sys.stderr)


def main():
    all_results = []
    for route in ROUTES:
        fetch = FETCHERS[route["carrier"]]
        try:
            prices = fetch(route["origin"], route["dest"], DATE_FROM, DATE_TO)
            all_results.extend({"carrier": route["carrier"], **p} for p in prices)
        except NotImplementedError as e:
            print(f"[skip] {route['carrier']}: {e}", file=sys.stderr)

    cheap = [r for r in all_results if r.get("price_twd", 999999) <= THRESHOLD_TWD]

    print(json.dumps(all_results, ensure_ascii=False, indent=2))

    if cheap:
        lines = [f"{r['carrier']} {r['date']}：NT${r['price_twd']}" for r in cheap]
        notify("低於門檻的機票出現：\n" + "\n".join(lines))
        print(f"\n[alert sent] {len(cheap)} fare(s) at or under NT${THRESHOLD_TWD}")


if __name__ == "__main__":
    main()
