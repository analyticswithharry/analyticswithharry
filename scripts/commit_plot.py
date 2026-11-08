import os, datetime as dt
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = "analyticswithharry"
START_YEAR = 2020
TOKEN = os.environ["GH_TOKEN"]

QUERY = """
query($login:String!, $from:DateTime!, $to:DateTime!) {
  user(login:$login) {
    contributionsCollection(from:$from, to:$to) {
      contributionCalendar {
        weeks { contributionDays { date contributionCount } }
      }
    }
  }
}
"""

def fetch_days():
    days = {}
    today = dt.date.today()
    for year in range(START_YEAR, today.year + 1):
        start = dt.datetime(year, 1, 1)
        end = min(dt.datetime(year, 12, 31, 23, 59, 59), dt.datetime.utcnow())
        r = requests.post(
            "https://api.github.com/graphql",
            json={"query": QUERY, "variables": {
                "login": USER,
                "from": start.isoformat() + "Z",
                "to": end.isoformat() + "Z"}},
            headers={"Authorization": f"bearer {TOKEN}"},
            timeout=60,
        )
        r.raise_for_status()
        cal = r.json()["data"]["user"]["contributionsCollection"]["contributionCalendar"]
        for w in cal["weeks"]:
            for d in w["contributionDays"]:
                days[d["date"]] = d["contributionCount"]
    return days

def weekly(days):
    buckets = {}
    for d, c in days.items():
        date = dt.date.fromisoformat(d)
        monday = date - dt.timedelta(days=date.weekday())
        buckets[monday] = buckets.get(monday, 0) + c
    xs = sorted(buckets)
    return xs, np.array([buckets[x] for x in xs])

def main():
    xs, ys = weekly(fetch_days())
    t = np.array([(x - xs[0]).days for x in xs], dtype=float)
    tn = t / t.max()

    coef = np.polyfit(tn, ys, 4)
    fit = np.poly1d(coef)
    ss_res = np.sum((ys - fit(tn)) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    grid = np.linspace(0, 1, 400)
    grid_dates = [xs[0] + dt.timedelta(days=float(g * t.max())) for g in grid]

    bg, blue, txt = "#0d1117", "#1E90FF", "#c9d1d9"
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    sizes = 14 + 70 * (ys / max(ys.max(), 1))
    ax.scatter(xs, ys, s=sizes, c=ys, cmap="Blues", alpha=0.85,
               edgecolors="none", zorder=3)
    ax.plot(grid_dates, np.clip(fit(grid), 0, None), color="#ff9f1c",
            lw=2.6, zorder=4, label="Polynomial regression (degree 4)")
    ax.fill_between(grid_dates, 0, np.clip(fit(grid), 0, None),
                    color=blue, alpha=0.08, zorder=2)

    ax.set_title("Weekly commit activity and non-linear trend",
                 color=txt, fontsize=14, pad=12, loc="left")
    ax.set_ylabel("Contributions per week", color=txt)
    ax.tick_params(colors=txt)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#30363d")
    ax.grid(color="#21262d", lw=0.6, zorder=0)
    ax.text(0.99, 0.95, f"R\u00b2 = {r2:.2f}", transform=ax.transAxes,
            ha="right", va="top", color=txt, fontsize=11)
    ax.legend(facecolor=bg, edgecolor="#30363d", labelcolor=txt, loc="upper left")

    os.makedirs("assets", exist_ok=True)
    fig.tight_layout()
    fig.savefig("assets/commit-trend.png", facecolor=bg)

if __name__ == "__main__":
    main()
