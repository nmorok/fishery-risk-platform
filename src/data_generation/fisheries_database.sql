-- ============================================
-- Fisheries and Disasters Database Schema
-- ============================================

--- Creating tables for reference database
CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE disaster_types (
    disaster_id INTEGER PRIMARY KEY,
    disaster_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE species (
    species_id INTEGER PRIMARY KEY,
    species_name VARCHAR(50) NOT NULL,
    region_id INTEGER NOT NULL,
    economic_value INEGER NOT NULL,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE species_disasters (
    species_id INTEGER NOT NULL,
    disaster_id INTEGER NOT NULL,
    vulnerability_score DEECIMAL(3,2) NOT NULL,
    PRIMARY KEY (species_id, disaster_id),
    FOREIGN KEY (species_id) REFERENCES species(species_id),
    FOREIGN KEY (disaster_id) REFERENCES disaster_types(disaster_id)
);

-- Insert data

INSERT INTO regions (region_id, region_name) VALUES
(1, 'Alaska'),
(2, 'Gulf of Mexico'),
(3, 'Northeast'),
(4, 'West Coast'), 
(5, 'Southeast');

INSERT INTO disaster_types (disaster_id, disaster_name) VALUES
(1, 'Warm Ocean Conditions'),
(2, 'Hurricane'),
(3, 'Harmful algal bloom'),
(4, 'Cold water event'),
(5, 'Flooding'),
(6, 'Oil spill');

-- ============================================
-- ALASKA Species
-- ============================================

INSERT INTO species (species_id, species_name, region_id, economic_value) VALUES
(1, 'Snow Crab', 1, 200),
(2, 'King Crab', 1, 150),
(3, 'Pacific Cod', 1, 300),
(4, 'Pacific Salmon', 1, 400), 
(5, 'Pollock', 1, 500);


INSERT INTO species_disasters (species_id, disaster_id, vulnerability_score) VALUES
-- Snow crab
(1, 1, 0.5), -- Warm ocean conditions
(1, 3, 0.3), -- Harmful algal bloom

-- King crab
(2, 1, 0.5), -- warm ocean conditions
(2, 3, 0.3), -- Harmful algal bloom

-- Pacific cod
(3, 1, 0.5), -- Warm ocean conditions
(3, 3, 0.3), -- Harmful algal bloom
(3, 4, 0.2), -- Cold water event

-- Pacific Salmon
(4, 1, 0.5),  -- Warm ocean conditions
(4, 3, 0.3),  -- Harmful algal bloom

-- Pollock
(5, 1, 0.4),  -- Warm ocean conditions
(5, 3, 0.2);  -- Harmful algal bloom

-- ============================================
-- WEST COAST Species
-- ============================================
INSERT INTO species (species_id, species_name, region_id, economic_value) VALUES
(6, 'Dungeness Crab', 2, 250),
(7, 'Pacific Salmon', 2, 350),
(8, 'Anchovy', 2, 100),
(9, 'Sardine', 2, 150),
(10, 'Albacore Tuna', 2, 200);

-- West Coast vulnerabilities
INSERT INTO species_disasters (species_id, disaster_id, vulnerability_score) VALUES
-- Dungeness Crab
(6, 1, 0.5),  -- Warm ocean conditions
(6, 3, 0.3),  -- Harmful algal bloom
-- Pacific Salmon (West Coast)
(7, 1, 0.5),  -- Warm ocean conditions
(7, 3, 0.3),  -- Harmful algal bloom
-- Anchovy
(8, 1, 0.4),  -- Warm ocean conditions
(8, 3, 0.2),  -- Harmful algal bloom
(8, 4, 0.3),  -- Cold water event
-- Sardine
(9, 1, 0.4),  -- Warm ocean conditions
(9, 3, 0.2),  -- Harmful algal bloom
(9, 4, 0.3),  -- Cold water event
-- Albacore Tuna
(10, 1, 0.5), -- Warm ocean conditions
(10, 3, 0.3); -- Harmful algal bloom

-- ============================================
-- GULF OF MEXICO Species
-- ============================================
INSERT INTO species (species_id, species_name, region_id, economic_value) VALUES
(11, 'Brown Shrimp', 3, 300),
(12, 'White Shrimp', 3, 250),
(13, 'Red Snapper', 3, 200),
(14, 'Oyster', 3, 150),
(15, 'Stone Crab', 3, 100);

-- Gulf of Mexico vulnerabilities
INSERT INTO species_disasters (species_id, disaster_id, vulnerability_score) VALUES
-- Brown Shrimp
(11, 2, 0.3), -- Hurricane
(11, 6, 0.4), -- Oil spill
(11, 5, 0.2), -- Flooding
-- White Shrimp
(12, 2, 0.3), -- Hurricane
(12, 6, 0.4), -- Oil spill
(12, 5, 0.2), -- Flooding
-- Red Snapper
(13, 2, 0.3), -- Hurricane
(13, 6, 0.4), -- Oil spill
-- Oyster
(14, 2, 0.3), -- Hurricane
(14, 6, 0.4), -- Oil spill
(14, 5, 0.2), -- Flooding
-- Stone Crab
(15, 2, 0.3), -- Hurricane
(15, 6, 0.4), -- Oil spill
(15, 5, 0.2); -- Flooding

-- ============================================
-- NORTHEAST Species
-- ============================================
INSERT INTO species (species_id, species_name, region_id, economic_value) VALUES
(16, 'American Lobster', 4, 600),
(17, 'Atlantic Cod', 4, 200),
(18, 'Sea Scallop', 4, 300),
(19, 'Atlantic Herring', 4, 150),
(20, 'Haddock', 4, 100);

-- Northeast vulnerabilities
INSERT INTO species_disasters (species_id, disaster_id, vulnerability_score) VALUES
-- American Lobster
(16, 1, 0.5), -- Warm ocean conditions
(16, 2, 0.3), -- Hurricane
(16, 4, 0.2), -- Cold water event
-- Atlantic Cod
(17, 1, 0.4), -- Warm ocean conditions
(17, 2, 0.3), -- Hurricane
(17, 4, 0.3), -- Cold water event
-- Sea Scallop
(18, 1, 0.4), -- Warm ocean conditions
(18, 2, 0.3), -- Hurricane
(18, 4, 0.2), -- Cold water event
(18, 3, 0.3), -- Harmful algal bloom
-- Atlantic Herring
(19, 1, 0.4), -- Warm ocean conditions
(19, 2, 0.3), -- Hurricane
(19, 4, 0.3), -- Cold water event
-- Haddock
(20, 1, 0.4), -- Warm ocean conditions
(20, 2, 0.3), -- Hurricane
(20, 4, 0.3); -- Cold water event

-- ============================================
-- SOUTHEAST Species
-- ============================================
INSERT INTO species (species_id, species_name, region_id, economic_value) VALUES
(21, 'Blue Crab', 5, 200),
(22, 'Red Drum', 5, 150),
(23, 'Spotted Seatrout', 5, 100),
(24, 'Spanish Mackerel', 5, 120),
(25, 'Black Sea Bass', 5, 130);

-- Southeast vulnerabilities
INSERT INTO species_disasters (species_id, disaster_id, vulnerability_score) VALUES
-- Blue Crab
(21, 2, 0.3), -- Hurricane
(21, 5, 0.4), -- Flooding
-- Red Drum
(22, 2, 0.3), -- Hurricane
-- Spotted Seatrout
(23, 2, 0.3), -- Hurricane
-- Spanish Mackerel
(24, 2, 0.3), -- Hurricane
-- Black Sea Bass
(25, 2, 0.3); -- Hurricane





-- ============================================
-- Practice Queries
-- ============================================

-- View all species with their regions and values

-- view species vulnerabilities

-- Find most vulnerable species to each disaster type


-- calculate total economic value at risk per region for each disaster