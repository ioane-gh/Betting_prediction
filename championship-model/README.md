# Championship BTTS + Over 2.5 engine

For every English Championship (EFL tier 2) fixture on a given day, this estimates:

- **P(both teams to score)**
- **P(total goals > 2.5)**

plus 1X2 as a free by-product, from a time-decayed Dixon-Coles bivariate Poisson
fit. Free data sources throughout; SQL Server for persistence; Python for
everything else.

---

## Calibration of expectations

Read this before reading any number the model prints.

- **Championship scoring sits near 2.5 goals a game**, so `P(over 2.5)` will
  cluster around **0.45-0.55** for most fixtures, and `P(BTTS)` likewise.
  Confident-looking extremes are rare, and usually mean an input is wrong
  rather than that a certainty has been found.
- **The closing line already contains team news, weather and money.** A free
  model that *matches* the de-vigged closing line is a genuine success. One that
  appears to beat it by ten percentage points has a bug — look at the inputs
  before looking at the bank balance.
- **This is a probability estimator, not a betting recommendation.** Bookmaker
  margins on BTTS and Over/Under run **4-7%**. Any real edge has to clear that
  margin before it exists at all, and the daily table's `Edge` column is
  computed against a de-vigged price precisely so that the margin is not
  mistaken for skill.

---

## Status of the numbers in this repository

The code is complete and tested across all nine phases. One thing is **not**
done, and cannot be done from the environment this was built in:

**No real Championship data has been through this pipeline yet.** The build
environment's egress policy blocks `football-data.co.uk`, `api.football-data.org`
and `fbref.com` (all three return `connect_rejected` from the proxy). So:

- every ingest path is implemented and unit-tested against fixtures, but has
  never seen a live `E1.csv`;
- the backtest numbers quoted below come from `champmodel.synth`, a **simulator**
  that generates seasons from known Dixon-Coles parameters and prices a
  synthetic closing line with a 5% overround. It exists so the machinery could
  be proved correct offline. It is not data, and no number derived from it says
  anything about the Championship;
- the tuned hyperparameters have therefore **not** been chosen on real data.
  `CHAMP_DECAY_HALF_LIFE_DAYS` is left at the literature-standard 180 days.

**First thing to do on a machine with network access:**

```powershell
champmodel init-db
champmodel ingest --backfill          # ~5,500 matches, 10 seasons
champmodel backtest --seasons 3       # the real acceptance test
champmodel tune --seasons 2           # then adopt the winning values in .env
```

(Setting up from scratch on Windows? Start at **Install** below and come back.)

Phase 7's acceptance bar — log loss within ~0.01 of the de-vigged closing line,
no calibration bucket off by more than a few points — must be met **on that
run** before any output is worth reading.

---

## Install (Windows / PowerShell)

Everything below is PowerShell on Windows. macOS and Linux equivalents are at
the end of this section.

You need **Python 3.10+** and, for the real database, **SQL Server 2019+** with
the **ODBC Driver 18**. Check what you have:

```powershell
python --version
# if that opens the Microsoft Store instead, use the launcher:
py -3 --version

# is an ODBC driver installed?
Get-OdbcDriver -Name "*SQL Server*" | Select-Object -ExpandProperty Name
```

If no ODBC driver is listed, install one (needed only for SQL Server — the
tests run without it):

```powershell
winget install --id Microsoft.msodbcsql.18
```

### Set the project up

```powershell
cd championship-model

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev,mssql]"        # add ,fbref for optional xG

Copy-Item .env.example .env
notepad .env                          # edit, see below
```

If activation fails with *"running scripts is disabled on this system"*, allow
signed local scripts for your user once — this does not need an admin prompt:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then `.\.venv\Scripts\Activate.ps1` again. Your prompt should read `(.venv)`.

Check the install:

```powershell
champmodel --help
pytest -q                             # 202 tests, ~19s, no database needed
```

`champmodel` is also runnable as `python -m champmodel.cli`, which is useful if
the `Scripts` directory is not on `PATH`.

### Configure `.env`

Set these for a local SQL Server using your Windows login:

```ini
CHAMP_DB_HOST=.\SQLEXPRESS
CHAMP_DB_NAME=champ
CHAMP_DB_TRUSTED_CONNECTION=1
CHAMP_DB_USER=
CHAMP_DB_PASSWORD=
CHAMP_DB_DRIVER=ODBC Driver 18 for SQL Server
FOOTBALL_DATA_ORG_KEY=your-free-key-here
```

`CHAMP_DB_HOST` takes whatever you would type into SSMS:

| your setup | `CHAMP_DB_HOST` |
|---|---|
| SQL Server Express | `.\SQLEXPRESS` or `localhost\SQLEXPRESS` |
| LocalDB (ships with SSMS / Visual Studio) | `(localdb)\MSSQLLocalDB` |
| Default instance on this machine | `localhost` |
| A server elsewhere | `dbhost.example.com` (add `CHAMP_DB_PORT` if not 1433) |

A named instance and LocalDB are resolved by instance name rather than by port,
so the port is left out of the connection string automatically — don't add one.

`.env` is gitignored. It holds your database password when you use SQL
authentication, so keep it that way.

Get the free football-data.org key at
<https://www.football-data.org/client/register> — it arrives by email in a
minute or two. The historical backfill does not need it; only today's fixtures
do.

### No SQL Server? Run on SQLite

The whole pipeline works on SQLite, which is how the tests run. Put this in
`.env` **instead of** the `CHAMP_DB_*` lines above and skip the ODBC install:

```ini
CHAMP_DB_URL=sqlite:///./data/champ.db
```

Everything below works identically. Move to SQL Server later by swapping the
`.env` lines and re-running `champmodel init-db` and the backfill.

You will see **two** files appear, `champ.db` and `champ.champ.db`. That is
expected, not a bug: SQL Server puts everything in a `champ` schema, and SQLite
reaches a schema through `ATTACH DATABASE`, which is a second file. Keeping the
two backends on one set of table definitions is what lets the test suite run
without SQL Server. Back up or delete both together.

### Create the schema

On SQL Server, create the **database** first — `init-db` connects to it, so it
has to exist before the schema can go in. Once, from SSMS or from PowerShell:

```powershell
sqlcmd -S ".\SQLEXPRESS" -E -Q "IF DB_ID('champ') IS NULL CREATE DATABASE champ"
```

(`-E` is Windows authentication; use `-U sa -P yourpassword` for SQL auth.)
On SQLite there is nothing to do — the file is created for you.

Then:

```powershell
champmodel init-db
```

That creates the `champ` schema, its tables, indexes and views, and seeds the
team registry. It is safe to re-run. If you would rather apply the DDL by hand:

```powershell
sqlcmd -S ".\SQLEXPRESS" -E -d champ -i sql\001_schema.sql
sqlcmd -S ".\SQLEXPRESS" -E -d champ -i sql\002_views.sql
```

`sql/001_schema.sql` is the authoritative DDL; the SQLAlchemy Core metadata in
`champmodel/db.py` mirrors it, and `tests/test_schema_parity.py` fails if the
two ever drift apart. The test suite runs against SQLite (with a `champ` schema
attached), so no SQL Server instance is needed to develop. Point
`CHAMP_TEST_DB_URL` at a real instance to exercise the `MERGE` path:

```powershell
$env:CHAMP_TEST_DB_URL = "mssql+pyodbc://@.\SQLEXPRESS/champ_test?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes"
pytest -q
Remove-Item Env:\CHAMP_TEST_DB_URL
```

Every load is an idempotent upsert on a natural key. Re-running a day's ingest
updates rows; it never duplicates them.

### First real run

In order, once:

```powershell
champmodel ingest --backfill          # ~5,500 matches, 10 seasons. Minutes.
champmodel backtest --seasons 3       # the acceptance test. See Phase 7 below.
champmodel tune --seasons 2           # optional: then adopt the winners in .env
champmodel fit                        # store the fitted model
```

Then, each day you want numbers:

```powershell
notepad data\availability.csv         # today's team news, if you have it
champmodel ingest --today
champmodel predict --date today
```

`predict` prints the table, writes `output\YYYY-MM-DD.csv`, and stores the run
in `champ.model_run` / `champ.prediction`.

**Do not skip the backtest.** Until it has run on real data, the model is
untuned and uncalibrated, and `predict` will flag every row `uncalibrated`.

### Running it daily without typing anything

Register a Scheduled Task that runs at 09:00 each day. Adjust the paths:

```powershell
$project = "C:\path\to\championship-model"
$action = New-ScheduledTaskAction `
    -Execute "$project\.venv\Scripts\python.exe" `
    -Argument "-m champmodel.cli predict --date today" `
    -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00am
Register-ScheduledTask -TaskName "champmodel daily" -Action $action -Trigger $trigger
```

The working directory matters: `.env` is read from wherever the command runs.

Check on it, or remove it:

```powershell
Get-ScheduledTaskInfo -TaskName "champmodel daily"
Get-Content .\logs\champmodel-*.jsonl -Tail 20
Unregister-ScheduledTask -TaskName "champmodel daily" -Confirm:$false
```

A scheduled run cannot know tonight's team news, so its rows will carry
`stale_team_news`. That is the point of the flag — treat those numbers as the
pre-team-news product, and re-run `predict` by hand once you have line-ups.

### Troubleshooting

| symptom | cause and fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`, then activate again |
| `Can't open lib 'ODBC Driver 18 for SQL Server'` | driver not installed: `winget install --id Microsoft.msodbcsql.18`, or set `CHAMP_DB_DRIVER` to a name from `Get-OdbcDriver` |
| `SSL Provider: certificate chain was issued by an authority that is not trusted` | a local instance with a self-signed certificate: keep `CHAMP_DB_TRUST_CERT=yes` |
| `Login failed for user ''` | `CHAMP_DB_TRUSTED_CONNECTION=1` for your Windows login, or set `CHAMP_DB_USER` / `CHAMP_DB_PASSWORD` for SQL auth |
| `server was not found or was not accessible` | wrong `CHAMP_DB_HOST`, or the SQL Browser service is stopped: `Get-Service MSSQL*, SQLBrowser` |
| `Cannot open database "champ" requested by the login` | create the database first, see **Create the schema** |
| `could not find 001_schema.sql` | installed non-editable; re-run `pip install -e ".[dev,mssql]"` or apply the DDL with `sqlcmd` |
| `python` opens the Microsoft Store | use `py -3` instead, or turn off the App Execution Alias in Settings |
| `champmodel : command not found` | virtualenv not activated, or use `python -m champmodel.cli` |
| ingest exits with code 2 | a team name did not resolve — the message names it. Add it to `ALIASES` in `champmodel/ingest/teams.py`. This is deliberate; see below |
| every row flagged `uncalibrated` | run `champmodel backtest` once to produce the calibration artifact |

### macOS and Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",mssql" for pyodbc, ",fbref" for xG
cp .env.example .env
champmodel init-db
```

Commands are otherwise identical. Set environment variables with
`export VAR=value` rather than `$env:VAR = "value"`.

---

## Data sources

| Source | Use | Auth | Notes |
|---|---|---|---|
| **football-data.co.uk** (`E1.csv`) | Historical results, 1993→. Goals, shots, SoT, corners, cards, **closing odds**. | none | Cached in `data/raw`; only the current season is refreshed. The odds columns are the benchmark, not decoration. |
| **football-data.org** (`/v4/competitions/ELC/matches`) | Today's fixtures, kickoff times, results backfill. | free key | Rate-limited to 10 req/min in-process, responses cached to disk by date range. |
| **FBref** via `soccerdata` | Optional: team xG, player minutes and cards. | none | Behind `CHAMP_ENABLE_FBREF`; 3-second crawl delay. The model works without it. |
| **`data/availability.csv`** | Injuries and suspensions. | none | Hand-edited. There is no reliable free injury API — see Phase 5. |

**Flashscore is deliberately not used.** It sits behind Cloudflare, serves
results through an obfuscated internal feed rather than clean HTML, breaks
constantly, and its terms prohibit automated extraction. Everything needed here
is available through sanctioned channels.

**If a source is unreachable the pipeline degrades and says so.** A blocked
download falls back to a stale cache with a warning; a missing fixture feed is
recorded; a missing input is written to `champ.prediction.missing_inputs` on
every affected row. Nothing is silently defaulted.

---

## Team name resolution

The one place a pipeline like this fails silently. `Sheffield Weds`,
`Sheff Weds` and `Sheffield Wednesday FC` must all be one `team_id`, or one club
becomes two half-strength teams and every rating built on it is wrong.

So resolution is strict: names are normalised (case, accents, punctuation, the
FC/AFC noise words), then looked up in an explicit alias table, and **an
unmapped name raises rather than warns**. `champmodel ingest` exits non-zero on
one. `tests/test_aliases.py` scans every CSV in `data/raw` and fails on any name
the registry cannot map, so a newly promoted club is caught at test time rather
than at 2am.

---

## The model

For a fixture between home `i` and away `j`:

```
lambda_home = exp(alpha_i + beta_j + gamma)
lambda_away = exp(alpha_j + beta_i)
```

`alpha` is attack strength, `beta` is defensive concession (lower is a meaner
defence), `gamma` is home advantage. The parametrisation has one redundant
direction, so the fit is anchored and the result re-centred at `mean(alpha) = 0`.

Three things do the real work:

**Time decay.** Each historical match is weighted `exp(-xi * days_ago)`, with
`xi` set from a half-life (default 180 days). **This is how current form enters
the model** — recent matches dominate the fit, and the strengths that come out
are already form-adjusted. There is deliberately no separate "last 5 games"
feature: it double-counts the same information and degrades calibration,
precisely in the situation (a team on a streak) where regression to the mean is
strongest. `champmodel/features/form.py` computes form **for display only**, and
says so at the top of the file.

**The tau correction** on the 0-0, 1-0, 0-1 and 1-1 cells. Independent Poissons
get the low-scoring cells wrong, and that is exactly the region where BTTS and
Over 2.5 are decided.

**Shrinkage.** A club with few weighted matches — a promoted side in August — is
pulled toward the league mean by a ridge penalty whose strength falls to zero
once it has enough history. A team should not top the attack table on the
strength of one 4-0 win.

Fitting is `scipy.optimize.minimize(method="L-BFGS-B")` on the weighted negative
log-likelihood, with an **analytic gradient** (checked against finite
differences in `tests/test_dixon_coles.py`). That makes a full fit take ~20ms
instead of ~1s, which is what makes the hundreds of refits in a walk-forward
backtest practical.

The probabilities come off an 11x11 score matrix with tau applied and
renormalised:

- `P(over 2.5)` = cells where `h + a >= 3`
- `P(BTTS)` = cells where `h >= 1 and a >= 1`
- `P(home/draw/away)` = lower triangle / diagonal / upper triangle

Home advantage is handled by `gamma` plus the separate home and away roles in
the fit — never as a post-hoc adjustment.

---

## Availability (Phase 5)

There is no free, reliable, machine-readable injury feed. So there are two
layers, and the manual one is authoritative.

**Layer 1 — `data/availability.csv`**, edited before a run and loaded into
`champ.availability_override`. Five minutes of team news is enough.

```csv
team,player_name,reason,minutes_share,is_attacker,is_defender,valid_from,valid_to,note,source_url
Sheffield Wednesday,A Striker,INJURY,0.42,1,0,2026-09-01,,hamstring,https://...
```

**Layer 2 — derived suspension risk**, when FBref ingestion is on: players at
the Championship yellow-card thresholds (5 / 10 / 15) or sent off recently.
Advisory only; a manual row for the same player always wins.

The adjustment is deliberately small, explicit and **bounded**:

```
missing_attack_share = SUM minutes_share of unavailable attackers (weight 1.0)
                     + SUM minutes_share of unavailable others    (weight 0.3)
alpha_adj = alpha - k * missing_attack_share            k = 0.5
```

with the mirror image on `beta` for missing defenders and goalkeepers, and the
whole move **clipped at ±0.25 in log space** (about a 22% swing in expected
goals). The cap is not optional: `minutes_share` is hand-entered, and the first
time someone types `4.5` instead of `0.45` an uncapped adjustment produces a
lambda no market would recognise. Loading rejects out-of-range values outright
rather than clipping them, so the typo surfaces instead of hiding; the cap is
the second line of defence. `availability_applied` is written to every
prediction row so the adjustment's effect can be measured later.

> The schema adds `is_defender` alongside the specified `is_attacker`, because
> the mirror-image adjustment to `beta` needs to know who the defenders are.

---

## Head-to-head (Phase 6)

Two clubs meet twice a season, so a ten-year window yields ~20 matches, most
played by squads that have since turned over completely. **Beyond the team
strengths the model already fits, the predictive signal is close to zero**, and
the literature is consistent on this.

So H2H is **displayed** — last 10 meetings, BTTS and Over 2.5 hit rates, in the
daily table, as context for a human — and the optional adjustment

```
lambda_adj = lambda * (1 + w * (h2h_rate - model_rate))     w <= 0.1
```

is **off by default** (`CHAMP_ENABLE_H2H_ADJUSTMENT=0`), clamped to `w <= 0.1`
however the config is set, and ignored for pairs with fewer than four meetings.

`champmodel backtest --h2h` measures it. On the synthetic league it improved
Over 2.5 log loss by 0.0008 — but the simulator's team ratings persist across
seasons by construction, which is exactly the structure that makes H2H look
informative, so **that result is an artifact and not the proof Phase 6 asks
for**. Do not enable it until the same comparison on real data shows a real
improvement. It probably will not.

---

## Backtesting (Phase 7)

`champmodel backtest` walks forward: for each matchday, fit on matches
**strictly before** that date, predict, record. The strict inequality is
asserted, not assumed — twice, once inside the loop and once over the finished
frame — and `tests/test_no_leakage.py` also feeds it a deliberately leaked frame
to check the assertion bites. Refits are reused for up to `--refit-every` days,
which never relaxes the rule (a reused fit was trained on data older still).

Scoring, per market:

- log loss and Brier score;
- **the de-vigged closing line as benchmark** — `AvgC>2.5` / `AvgC<2.5`
  converted to implied probabilities with the overround removed
  proportionally, and scored on exactly the rows it priced so the comparison is
  like with like;
- a 10-bucket calibration table (predicted vs observed, with counts).

If the reliability curve is off, Platt scaling is fitted on an **earlier**
half of the backtest and scored on the **later** half, and refuses to fit at all
on fewer than 200 rows — an identity calibrator is more honest than two
coefficients fitted to noise. The coefficients are stored with the model run.

`champmodel tune` grid-searches the decay half-life and shrinkage prior on
**out-of-sample** log loss, never on training fit.

### What the synthetic run shows

Again: this is the simulator, not the Championship. It demonstrates that the
machinery works end to end. All four rows below are scored on the **same**
held-out later half of the backtest, so raw and calibrated are like for like.

| | model LL | market LL | gap |
|---|---|---|---|
| BTTS, raw | 0.6815 | 0.6819 | -0.0004 |
| BTTS, Platt-calibrated | 0.6827 | 0.6819 | +0.0008 |
| Over 2.5, raw | 0.6709 | 0.6658 | +0.0051 |
| Over 2.5, Platt-calibrated | 0.6681 | 0.6658 | +0.0023 |

Calibration halved the worst populated Over 2.5 bucket error, 0.152 to 0.088,
and closed half the log-loss gap. It left BTTS very slightly worse, which is
the honest result: the raw BTTS curve was already well calibrated, and there
was nothing for Platt scaling to fix.

The model tracks the synthetic closing line and does not beat it. That is the
correct outcome — the simulated market is priced off the true probabilities
plus noise, so beating it would mean something was wrong.

Note that `champmodel backtest` prints its *raw* report over the whole test
window and its *calibrated* report over the held-out half only, since the
earlier half was spent fitting the calibrator. The two blocks are therefore not
directly comparable; the table above is.

---

## Daily run (Phase 8)

```powershell
champmodel predict --date today
```

1. ingest today's fixtures (cached, rate-limited);
2. load availability overrides valid today;
3. load the stored fit, or refit if older than `CHAMP_REFIT_AFTER_DAYS`;
4. lambdas → availability → optional H2H → score matrix → calibration;
5. write `champ.model_run` + `champ.prediction`;
6. print the table and write `output/YYYY-MM-DD.csv`.

```
Kickoff | Home | Away | P(BTTS) | P(O2.5) | lH | lA | H2H BTTS | H2H O2.5 | Mkt O2.5 | Edge | Avail? | Flags
```

sorted by absolute edge descending, with fixtures the market has not priced
last.

**Every row shows its `missing_inputs`.** A prediction made without today's
team news is not the same product as one made with it, and the output says
which it is. The flags are:

| flag | meaning |
|---|---|
| `no_team_news` | no `availability.csv` at all |
| `stale_team_news` | the file was last edited before the fixture date |
| `no_market_odds` | no closing price to compare against |
| `no_h2h` | these clubs have no meetings on record |
| `uncalibrated` | no calibration artifact — run `champmodel backtest` |
| `home_team_unfitted` / `away_team_unfitted` | club not in the training window; league-mean ratings used |
| `home_team_shrunk` / `away_team_shrunk` | too few weighted matches; shrunk toward the mean |
| `home_avail_capped` / `away_avail_capped` | the availability adjustment hit its bound — check `minutes_share` |

---

## Commands

| | |
|---|---|
| `champmodel init-db` | create the schema, seed the team registry |
| `champmodel ingest --backfill` | 10 seasons of results |
| `champmodel ingest --today` | today's fixtures + availability |
| `champmodel fit` | fit and store the model |
| `champmodel backtest` | walk-forward vs the closing line |
| `champmodel tune` | grid-search hyperparameters |
| `champmodel predict --date today` | the daily run |
| `champmodel status` | row counts, stored fit, alias gate |

`--synthetic` on `backtest` and `tune` runs against generated data with a loud
warning, for checking the machinery without a database.

The commands are the same in PowerShell, cmd and bash. Only the setup steps
differ, and `--help` works on every one:

```powershell
champmodel predict --help
```

---

## Configuration

All of `.env` is documented in `.env.example`. The ones that matter:

| variable | default | |
|---|---|---|
| `CHAMP_DECAY_HALF_LIFE_DAYS` | 180 | how fast old matches stop counting — **tune on real data** |
| `CHAMP_MAX_GOALS` | 10 | score matrix size; 10 truncates ~3e-6 of the answer |
| `CHAMP_MIN_WEIGHTED_MATCHES` | 20 | below this a team is shrunk toward the mean |
| `CHAMP_SHRINKAGE_PRIOR` | 0.30 | how hard |
| `CHAMP_AVAILABILITY_K` | 0.5 | availability strength |
| `CHAMP_AVAILABILITY_CAP` | 0.25 | the bound. Do not remove it |
| `CHAMP_ENABLE_H2H_ADJUSTMENT` | 0 | leave off until proven on real data |
| `CHAMP_REFIT_AFTER_DAYS` | 7 | staleness before `predict` refits |

---

## Tests

```powershell
pytest -q          # 202 tests, ~19s. No database or network needed.
```

| file | what it guards |
|---|---|
| `test_score_matrix.py` | matrix sums to 1.0; with tau off, `P(over 2.5)` matches a closed-form independent Poisson |
| `test_no_leakage.py` | the training window never reaches the target match |
| `test_aliases.py` | every name in every raw CSV resolves; the Sheffield Wednesday spellings share one id |
| `test_idempotent.py` | running the same ingest twice leaves row counts unchanged |
| `test_dixon_coles.py` | analytic gradient vs finite differences; parameter recovery against known truth; shrinkage bites |
| `test_features.py` | the availability cap holds against a fat-fingered `minutes_share`; H2H weight is clamped |
| `test_metrics.py` | de-vigging, scoring rules, calibration tables |
| `test_pipeline.py` | the `missing_inputs` contract |
| `test_cli.py` | the daily run end to end |
| `test_schema_parity.py` | the DDL and the Python metadata describe the same tables |
| `test_db.py` | schema applies, a match round-trips |

Logging is structured JSON to `logs/champmodel-YYYYMMDD.jsonl`, with the
`run_id` on every record.

---

## Layout

```
championship-model/
  sql/001_schema.sql, 002_views.sql     authoritative DDL + reporting views
  src/champmodel/
    config.py db.py repository.py       config, SQLAlchemy Core, queries
    logging_setup.py synth.py           JSON logging; offline simulator
    ingest/   teams.py seasons.py footballdata_uk.py footballdata_org.py
              fbref.py availability.py
    features/ form.py h2h.py availability_adj.py
    model/    dixon_coles.py score_matrix.py calibrate.py pipeline.py
    backtest/ walk_forward.py metrics.py
    cli.py
  tests/  data/raw/ (gitignored)  output/
```
