/* ===========================================================================
   champ schema -- English Championship BTTS / Over 2.5 engine
   Target: SQL Server 2019+
   Idempotent: safe to run repeatedly against the same database.
   =========================================================================== */

IF SCHEMA_ID(N'champ') IS NULL
    EXEC(N'CREATE SCHEMA champ');
GO

/* --------------------------------------------------------------------------
   Teams and the source-specific name variants that map onto them.
   -------------------------------------------------------------------------- */
IF OBJECT_ID(N'champ.dim_team', N'U') IS NULL
CREATE TABLE champ.dim_team
(
    team_id         INT IDENTITY(1,1)   NOT NULL CONSTRAINT PK_dim_team PRIMARY KEY,
    canonical_name  NVARCHAR(100)       NOT NULL,
    fbref_id        NVARCHAR(20)        NULL,
    fd_org_id       INT                 NULL,
    CONSTRAINT UQ_dim_team_canonical UNIQUE (canonical_name)
);
GO

IF OBJECT_ID(N'champ.team_alias', N'U') IS NULL
CREATE TABLE champ.team_alias
(
    alias    NVARCHAR(100) NOT NULL CONSTRAINT PK_team_alias PRIMARY KEY,
    team_id  INT           NOT NULL CONSTRAINT FK_team_alias_team
                 REFERENCES champ.dim_team (team_id),
    source   NVARCHAR(30)  NOT NULL
);
GO

IF OBJECT_ID(N'champ.dim_season', N'U') IS NULL
CREATE TABLE champ.dim_season
(
    season_id  INT         NOT NULL CONSTRAINT PK_dim_season PRIMARY KEY,  -- 2526
    label      NVARCHAR(9) NOT NULL,                                       -- '2025-26'
    start_date DATE        NULL,
    end_date   DATE        NULL
);
GO

/* --------------------------------------------------------------------------
   Matches. home_goals IS NULL means the match has not been played yet.
   -------------------------------------------------------------------------- */
IF OBJECT_ID(N'champ.fact_match', N'U') IS NULL
CREATE TABLE champ.fact_match
(
    match_id     BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_fact_match PRIMARY KEY,
    season_id    INT          NOT NULL CONSTRAINT FK_fact_match_season
                     REFERENCES champ.dim_season (season_id),
    match_date   DATE         NOT NULL,
    kickoff_utc  DATETIME2(0) NULL,
    home_team_id INT          NOT NULL CONSTRAINT FK_fact_match_home
                     REFERENCES champ.dim_team (team_id),
    away_team_id INT          NOT NULL CONSTRAINT FK_fact_match_away
                     REFERENCES champ.dim_team (team_id),
    home_goals   TINYINT      NULL,
    away_goals   TINYINT      NULL,
    home_ht      TINYINT      NULL,
    away_ht      TINYINT      NULL,
    status       NVARCHAR(20) NOT NULL CONSTRAINT DF_fact_match_status DEFAULT (N'SCHEDULED'),
    source       NVARCHAR(30) NOT NULL,
    CONSTRAINT UQ_fact_match_natural UNIQUE (match_date, home_team_id, away_team_id),
    CONSTRAINT CK_fact_match_teams CHECK (home_team_id <> away_team_id),
    CONSTRAINT CK_fact_match_status
        CHECK (status IN (N'SCHEDULED', N'FINISHED', N'POSTPONED', N'IN_PLAY', N'CANCELLED'))
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'IX_fact_match_date' AND object_id = OBJECT_ID(N'champ.fact_match'))
    CREATE INDEX IX_fact_match_date ON champ.fact_match (match_date)
        INCLUDE (home_team_id, away_team_id, home_goals, away_goals, status);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'IX_fact_match_teams' AND object_id = OBJECT_ID(N'champ.fact_match'))
    CREATE INDEX IX_fact_match_teams ON champ.fact_match (home_team_id, away_team_id)
        INCLUDE (match_date, home_goals, away_goals);
GO

/* Optional match detail, 1:1 with fact_match. */
IF OBJECT_ID(N'champ.fact_match_stats', N'U') IS NULL
CREATE TABLE champ.fact_match_stats
(
    match_id     BIGINT   NOT NULL CONSTRAINT PK_fact_match_stats PRIMARY KEY
                     CONSTRAINT FK_fact_match_stats_match
                     REFERENCES champ.fact_match (match_id),
    home_shots   SMALLINT NULL,
    away_shots   SMALLINT NULL,
    home_sot     SMALLINT NULL,
    away_sot     SMALLINT NULL,
    home_corners SMALLINT NULL,
    away_corners SMALLINT NULL,
    home_yellow  SMALLINT NULL,
    away_yellow  SMALLINT NULL,
    home_red     SMALLINT NULL,
    away_red     SMALLINT NULL,
    home_xg      DECIMAL(5,2) NULL,
    away_xg      DECIMAL(5,2) NULL
);
GO

/* Closing odds. One row per (match, book); 'Avg' is the market-average
   column family from football-data.co.uk and is the honest benchmark. */
IF OBJECT_ID(N'champ.market_odds', N'U') IS NULL
CREATE TABLE champ.market_odds
(
    match_id       BIGINT       NOT NULL CONSTRAINT FK_market_odds_match
                       REFERENCES champ.fact_match (match_id),
    book           NVARCHAR(20) NOT NULL,
    odds_over25    DECIMAL(6,2) NULL,
    odds_under25   DECIMAL(6,2) NULL,
    odds_btts_yes  DECIMAL(6,2) NULL,
    odds_btts_no   DECIMAL(6,2) NULL,
    odds_home      DECIMAL(6,2) NULL,
    odds_draw      DECIMAL(6,2) NULL,
    odds_away      DECIMAL(6,2) NULL,
    CONSTRAINT PK_market_odds PRIMARY KEY (match_id, book)
);
GO

/* --------------------------------------------------------------------------
   Availability overrides (Phase 5). Hand-maintained; authoritative.
   valid_to IS NULL means "still out".
   -------------------------------------------------------------------------- */
IF OBJECT_ID(N'champ.availability_override', N'U') IS NULL
CREATE TABLE champ.availability_override
(
    override_id   INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_availability_override PRIMARY KEY,
    team_id       INT           NOT NULL CONSTRAINT FK_availability_team
                      REFERENCES champ.dim_team (team_id),
    player_name   NVARCHAR(100) NOT NULL,
    reason        NVARCHAR(20)  NOT NULL,
    minutes_share DECIMAL(5,4)  NOT NULL,
    is_attacker   BIT           NOT NULL CONSTRAINT DF_availability_attacker DEFAULT (0),
    is_defender   BIT           NOT NULL CONSTRAINT DF_availability_defender DEFAULT (0),
    valid_from    DATE          NOT NULL,
    valid_to      DATE          NULL,
    note          NVARCHAR(400) NULL,
    source_url    NVARCHAR(400) NULL,
    CONSTRAINT CK_availability_reason
        CHECK (reason IN (N'INJURY', N'SUSPENSION', N'DOUBT')),
    CONSTRAINT CK_availability_share
        CHECK (minutes_share >= 0 AND minutes_share <= 1),
    CONSTRAINT UQ_availability_natural UNIQUE (team_id, player_name, valid_from)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'IX_availability_window'
                 AND object_id = OBJECT_ID(N'champ.availability_override'))
    CREATE INDEX IX_availability_window
        ON champ.availability_override (team_id, valid_from, valid_to);
GO

/* --------------------------------------------------------------------------
   Model runs and their predictions.
   -------------------------------------------------------------------------- */
IF OBJECT_ID(N'champ.model_run', N'U') IS NULL
CREATE TABLE champ.model_run
(
    run_id         BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_model_run PRIMARY KEY,
    run_utc        DATETIME2(3)  NOT NULL CONSTRAINT DF_model_run_utc DEFAULT (SYSUTCDATETIME()),
    model_version  NVARCHAR(20)  NOT NULL,
    params         NVARCHAR(MAX) NULL,      -- JSON
    train_rows     INT           NULL,
    train_end_date DATE          NULL
);
GO

IF OBJECT_ID(N'champ.prediction', N'U') IS NULL
CREATE TABLE champ.prediction
(
    prediction_id        BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_prediction PRIMARY KEY,
    run_id               BIGINT       NOT NULL CONSTRAINT FK_prediction_run
                             REFERENCES champ.model_run (run_id),
    match_id             BIGINT       NOT NULL CONSTRAINT FK_prediction_match
                             REFERENCES champ.fact_match (match_id),
    lambda_home          DECIMAL(6,4) NOT NULL,
    lambda_away          DECIMAL(6,4) NOT NULL,
    p_btts               DECIMAL(6,5) NOT NULL,
    p_over25             DECIMAL(6,5) NOT NULL,
    p_home               DECIMAL(6,5) NULL,
    p_draw               DECIMAL(6,5) NULL,
    p_away               DECIMAL(6,5) NULL,
    availability_applied BIT          NOT NULL CONSTRAINT DF_prediction_avail DEFAULT (0),
    missing_inputs       NVARCHAR(200) NULL,
    CONSTRAINT UQ_prediction_run_match UNIQUE (run_id, match_id)
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = N'IX_prediction_match' AND object_id = OBJECT_ID(N'champ.prediction'))
    CREATE INDEX IX_prediction_match ON champ.prediction (match_id);
GO
