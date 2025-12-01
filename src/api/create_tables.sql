-- ============================================
-- CORE ENTITIES
-- ============================================

-- Disaster declarations from NOAA
-- ex: 3 | 1994-03-15 | "West Coast Salmon failure due to ocean conditions"
--     6 | 1995-01-20 | "Continuation of disaster #3"
CREATE TABLE disasters (
    disaster_id INTEGER PRIMARY KEY, -- this matches the NOAA disaster number
    request_date DATE, -- the date the disaster was requested by fishery managers
    determination_date DATE, -- the date the disaster was declared by NOAA
    title TEXT -- Title of the disaster from the NOAA website
);

-- Parent-child disaster relationships (continuations)
-- ex: 3 | 6  (disaster 3 continued as disaster 6)
--     7 | 11  (disaster 7 continued as disaster 11)
CREATE TABLE disaster_relationships (
    parent_disaster_id INTEGER, -- the original disaster
    child_disaster_id INTEGER, -- the continuation disaster if it exists
    PRIMARY KEY (parent_disaster_id, child_disaster_id),
    FOREIGN KEY (parent_disaster_id) REFERENCES disasters(disaster_id),
    FOREIGN KEY (child_disaster_id) REFERENCES disasters(disaster_id)
);

-- Years affected by each disaster
-- ex: 3 | 1992
--     3 | 1993
--     3 | 1994
--     6 | 1995
--     7 | 1997
--     7 | 1998
CREATE TABLE disaster_years (
    disaster_id INTEGER, -- the disaster this year is associated with
    year INTEGER, -- the affected year
    PRIMARY KEY (disaster_id, year),
    FOREIGN KEY (disaster_id) REFERENCES disasters(disaster_id)
);

-- Species reference
-- ex: 1 | "Chinook" | "Salmon"
--     2 | "Coho" | "Salmon"
--     3 | "Sockeye" | "Salmon"
--     4 | "Dungeness Crab" | "Crab"
--     5 | "Pacific Cod" | "Pacific Cod"
CREATE TABLE species (
    species_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique identifier for each species
    species_name VARCHAR(100) UNIQUE NOT NULL, -- Common name of the species
    species_family VARCHAR(100) -- Family or group the species belongs to
);

-- ============================================
-- FISHERY EVENT DATA
-- ============================================

-- Main fishery event records (one row from your spreadsheet)
-- ex: 1 | 'West_coast' | 'West Coast Salmon' | 'West Coast' | 3 | 2 | 4 | 74743831.00 | 61092570.00 | 1992
--     2 | 'Alaska' | 'Bristol Bay Salmon' | 'Alaska' | 4 | 5 | 4 | 824751476.00 | 152896270.00 | 1997
--     8 | 'West_coast' | 'California Dungeness and Rock Crab' | 'California' | 2 | 2 | 1 | 97929007.00 | 33896710.00 | NULL

CREATE TABLE fishery_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique identifier for each fishery event
    management_zone VARCHAR(50), -- The management zone for the fishery event
    sst_region VARCHAR(100), -- The spatial mask for the sst data
    fishery_name VARCHAR(255), -- The name of the fishery
    num_fisheries INTEGER, -- Number of fisheries affected
    num_species INTEGER, -- Number of species affected
    num_years INTEGER, -- Number of years affected
    total_value REAL, -- Total value of the fishery in Feb 2025 $
    appropriation REAL, -- Appropriation amount in Feb 2025 $
    sst_year INTEGER  -- unknown if needed right now
);


-- SST anomalies metadata for each disaster
-- ex: 3 | 'Washington_BC' | 1992-04-01 | 1994-09-30 | TRUE
-- ex 47 | 'California' | 2015-01-01 | 2015-12-31 | FALSE
CREATE TABLE sst_anomalies_metadata (
    disaster_id INTEGER PRIMARY KEY, 
    shapefile VARCHAR(255), -- the shapefile used for the spatial masking when calculating SST anomalies
    fishery_start DATE, -- start date for fishery event SST anomaly calculation
    fishery_end DATE, -- end date for fishery event SST anomaly calculation
    used_start INTEGER, -- whether the start date was used (1) or the end date (0) - some disasters happened during the fishery season, so we use the end of the fishery for the SST calculation
    FOREIGN KEY (disaster_id) REFERENCES disasters(disaster_id)
);

-- Link events to disasters (many-to-many)
-- ex: 1 | 3  (event 1 is linked to disaster 3)
--     1 | 6  (event 1 is also linked to disaster 6)
--     2 | 7
--     2 | 11
--     2 | 18
CREATE TABLE event_disasters (
    event_id INTEGER, -- the fishery event
    disaster_id INT, -- the linked disaster
    PRIMARY KEY (event_id, disaster_id),
    FOREIGN KEY (event_id) REFERENCES fishery_events(event_id),
    FOREIGN KEY (disaster_id) REFERENCES disasters(disaster_id)
);

-- Link events to species (many-to-many)
-- ex: 1 | 1  (event 1 affected Chinook)
--     1 | 2  (event 1 affected Coho)
--     2 | 1  (event 2 affected Chinook)
--     2 | 2  (event 2 affected Coho)
--     2 | 3  (event 2 affected Sockeye)
--     8 | 4  (event 8 affected Dungeness Crab)
CREATE TABLE event_species (
    event_id INTEGER, -- the fishery event
    species_id INTEGER, -- the affected species
    PRIMARY KEY (event_id, species_id),
    FOREIGN KEY (event_id) REFERENCES fishery_events(event_id),
    FOREIGN KEY (species_id) REFERENCES species(species_id)
);

-- Link events to affected years (many-to-many)
-- ex: 1 | 1992
--     1 | 1993
--     1 | 1994
--     1 | 1995
--     2 | 1997
--     2 | 1998
--     2 | 1999
--     2 | 2000
CREATE TABLE event_years (
    event_id INTEGER, -- the fishery event
    year INTEGER, -- the affected year
    PRIMARY KEY (event_id, year),
    FOREIGN KEY (event_id) REFERENCES fishery_events(event_id)
);

-- ============================================
-- ANALYSIS DATA
-- ============================================

-- Heatwave metrics for each event (one heatwave metric per event)
-- ex: 1 | 2.35 | 45 | 123.45 | 2.74 | 1992-05-15
--     2 | 3.12 | 67 | 209.04 | 3.12 | 1997-06-20
--     8 | 4.87 | 234 | 1139.58 | 4.87 | 2015-03-01
CREATE TABLE heatwave_metrics (
    event_id INTEGER PRIMARY KEY,
    region TEXT,
    peak_intensity REAL,
    duration_days REAL, -- renamed from sum_duration
    mean_duration REAL,
    median_duration REAL,
    sum_cumulative_intensity REAL,
    mean_cumulative_intensity REAL,
    median_cumulative_intensity REAL,
    percent_in_heatwave REAL, -- renamed from percent_area_in_heatwave
    total_cells INTEGER,
    cells_with_heatwave INTEGER,
    total_events INTEGER,
    FOREIGN KEY (event_id) REFERENCES fishery_events(event_id)
);


-- optional indexes to speed up queries
CREATE INDEX idx_event_years_year ON event_years(year);
CREATE INDEX idx_disaster_years_year ON disaster_years(year);
CREATE INDEX idx_fishery_events_zone ON fishery_events(management_zone);
CREATE INDEX idx_fishery_events_sst_region ON fishery_events(sst_region);
CREATE INDEX idx_event_species_species_id ON event_species(species_id);
CREATE INDEX idx_species_family ON species(species_family);