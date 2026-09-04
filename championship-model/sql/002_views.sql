/* ===========================================================================
   champ views -- reporting and data-quality helpers
   Run after 001_schema.sql.
   =========================================================================== */

/* Every finished match with readable team names and the derived market
   outcomes. This is the view the model trains from. */
CREATE OR ALTER VIEW champ.vw_match_result AS
SELECT
    m.match_id,
    m.season_id,
    s.label            AS season_label,
    m.match_date,
    m.kickoff_utc,
    m.home_team_id,
    ht.canonical_name  AS home_team,
    m.away_team_id,
    at.canonical_name  AS away_team,
    m.home_goals,
    m.away_goals,
    m.home_goals + m.away_goals AS total_goals,
    CAST(CASE WHEN m.home_goals >= 1 AND m.away_goals >= 1 THEN 1 ELSE 0 END AS BIT) AS btts,
    CAST(CASE WHEN m.home_goals + m.away_goals >= 3 THEN 1 ELSE 0 END AS BIT)        AS over25,
    m.status,
    m.source
FROM champ.fact_match  AS m
JOIN champ.dim_team    AS ht ON ht.team_id = m.home_team_id
JOIN champ.dim_team    AS at ON at.team_id = m.away_team_id
JOIN champ.dim_season  AS s  ON s.season_id = m.season_id
WHERE m.status = N'FINISHED'
  AND m.home_goals IS NOT NULL
  AND m.away_goals IS NOT NULL;
GO

/* Closing market-average Over/Under 2.5 with the overround removed
   proportionally. This is the benchmark the model is scored against. */
CREATE OR ALTER VIEW champ.vw_market_over25_devig AS
SELECT
    o.match_id,
    o.book,
    o.odds_over25,
    o.odds_under25,
    CAST((1.0 / o.odds_over25) / ((1.0 / o.odds_over25) + (1.0 / o.odds_under25))
         AS DECIMAL(6,5)) AS p_over25_devig,
    CAST((1.0 / o.odds_over25) + (1.0 / o.odds_under25) AS DECIMAL(6,5)) AS overround
FROM champ.market_odds AS o
WHERE o.odds_over25 > 1.0
  AND o.odds_under25 > 1.0;
GO

/* Same treatment for BTTS, where the source has the columns. */
CREATE OR ALTER VIEW champ.vw_market_btts_devig AS
SELECT
    o.match_id,
    o.book,
    o.odds_btts_yes,
    o.odds_btts_no,
    CAST((1.0 / o.odds_btts_yes) / ((1.0 / o.odds_btts_yes) + (1.0 / o.odds_btts_no))
         AS DECIMAL(6,5)) AS p_btts_devig,
    CAST((1.0 / o.odds_btts_yes) + (1.0 / o.odds_btts_no) AS DECIMAL(6,5)) AS overround
FROM champ.market_odds AS o
WHERE o.odds_btts_yes > 1.0
  AND o.odds_btts_no > 1.0;
GO

/* Data-quality gate (Phase 3c): any alias that does not resolve to a team.
   The ingest asserts this returns zero rows. */
CREATE OR ALTER VIEW champ.vw_unresolved_alias AS
SELECT a.alias, a.source, a.team_id
FROM champ.team_alias AS a
LEFT JOIN champ.dim_team AS t ON t.team_id = a.team_id
WHERE t.team_id IS NULL;
GO

/* Latest prediction per match, with the market comparison and the edge. */
CREATE OR ALTER VIEW champ.vw_latest_prediction AS
WITH ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (PARTITION BY p.match_id ORDER BY r.run_utc DESC, p.run_id DESC) AS rn
    FROM champ.prediction AS p
    JOIN champ.model_run  AS r ON r.run_id = p.run_id
)
SELECT
    r.match_id,
    r.run_id,
    m.match_date,
    m.kickoff_utc,
    ht.canonical_name AS home_team,
    at.canonical_name AS away_team,
    r.lambda_home,
    r.lambda_away,
    r.p_btts,
    r.p_over25,
    r.p_home,
    r.p_draw,
    r.p_away,
    r.availability_applied,
    r.missing_inputs,
    d.p_over25_devig  AS market_p_over25,
    CAST(r.p_over25 - d.p_over25_devig AS DECIMAL(6,5)) AS edge_over25
FROM ranked AS r
JOIN champ.fact_match AS m  ON m.match_id = r.match_id
JOIN champ.dim_team   AS ht ON ht.team_id = m.home_team_id
JOIN champ.dim_team   AS at ON at.team_id = m.away_team_id
LEFT JOIN champ.vw_market_over25_devig AS d
       ON d.match_id = r.match_id AND d.book = N'Avg'
WHERE r.rn = 1;
GO

/* Rolling head-to-head record, used for the display columns in Phase 6. */
CREATE OR ALTER VIEW champ.vw_h2h_pair AS
SELECT
    CASE WHEN m.home_team_id < m.away_team_id THEN m.home_team_id ELSE m.away_team_id END AS team_a_id,
    CASE WHEN m.home_team_id < m.away_team_id THEN m.away_team_id ELSE m.home_team_id END AS team_b_id,
    m.match_id,
    m.match_date,
    m.home_team_id,
    m.away_team_id,
    m.home_goals,
    m.away_goals
FROM champ.fact_match AS m
WHERE m.status = N'FINISHED'
  AND m.home_goals IS NOT NULL
  AND m.away_goals IS NOT NULL;
GO
