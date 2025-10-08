"""Reference data for fisheries and disasters in US regions"""

class ReferenceData:
    """Reference data for fisheries and disasters in US regions"""

    # Species available in each region
    # access by ReferenceData.SPECIES_BY_REGION['Alaska'], etc.
    SPECIES_BY_REGION = {
        'Alaska': [
            'Snow Crab', 'King Crab', 'Pacific Cod', 
            'Pacific Salmon', 'Pollock'
        ],
        'West Coast': [
            'Dungeness Crab', 'Pacific Salmon', 'Anchovy', 
            'Sardine', 'Albacore Tuna'
        ],
        'Gulf of Mexico': [
            'Brown Shrimp', 'White Shrimp', 'Red Snapper', 
            'Oyster', 'Stone Crab'
        ],
        'Northeast': [
            'American Lobster', 'Atlantic Cod', 'Sea Scallop',
            'Atlantic Herring', 'Haddock'
        ],
        'Southeast': [
            'Blue Crab', 'Red Drum', 'Spotted Seatrout',
            'Spanish Mackerel', 'Black Sea Bass'
        ]
    }

    # Disaster types
    # access by ReferenceData.DISASTER_TYPES, etc.
    DISASTER_TYPES = [
        'Warm ocean conditions',
        'Hurricane',
        'Harmful algal bloom', 
        'Cold water event',
        'Flooding',
        'Oil spill'
    ]

    # ========== ALASKA ==========
    # access by ReferenceData.ALASKA_SPECIES_VALUE, etc.
    ALASKA_SPECIES_VALUE = {
        'Snow Crab': 200,
        'King Crab': 150,
        'Pacific Cod': 300,
        'Pacific Salmon': 400,
        'Pollock': 500
    }

    # access by ReferenceData.ALASKA_SPECIES_DISASTERS, etc.
    ALASKA_SPECIES_DISASTERS = {
        'Snow Crab': ['Warm ocean conditions', 'Harmful algal bloom'],
        'King Crab': ['Warm ocean conditions', 'Harmful algal bloom'],
        'Pacific Cod': ['Warm ocean conditions', 'Cold water event', 'Harmful algal bloom'],
        'Pacific Salmon': ['Warm ocean conditions', 'Harmful algal bloom'],
        'Pollock': ['Warm ocean conditions', 'Harmful algal bloom']
    }

    # access by ReferenceData.ALASKA_VULNERABILITY, etc.
    ALASKA_VULNERABILITY = {
        'Snow Crab': {
            'Warm ocean conditions': 0.5,
            'Harmful algal bloom': 0.3
        },
        'King Crab': {
            'Warm ocean conditions': 0.6,
            'Harmful algal bloom': 0.4
        },
        'Pacific Cod': {
            'Warm ocean conditions': 0.4,
            'Cold water event': 0.2,
            'Harmful algal bloom': 0.3
        },
        'Pacific Salmon': {
            'Warm ocean conditions': 0.5,
            'Harmful algal bloom': 0.3
        },
        'Pollock': {
            'Warm ocean conditions': 0.4,
            'Harmful algal bloom': 0.2
        }
    }

    # ========== WEST COAST ==========
    WEST_COAST_SPECIES_VALUE = {
        'Dungeness Crab': 250,
        'Pacific Salmon': 350,
        'Anchovy': 100,
        'Sardine': 150,
        'Albacore Tuna': 200
    }
    
    WEST_COAST_SPECIES_DISASTERS = {
        'Dungeness Crab': ['Warm ocean conditions', 'Harmful algal bloom'],
        'Pacific Salmon': ['Warm ocean conditions', 'Harmful algal bloom'],
        'Anchovy': ['Warm ocean conditions', 'Harmful algal bloom', 'Cold water event'],
        'Sardine': ['Warm ocean conditions', 'Harmful algal bloom', 'Cold water event'],
        'Albacore Tuna': ['Warm ocean conditions', 'Harmful algal bloom']
    }
    
    WEST_COAST_VULNERABILITY = {
        'Dungeness Crab': {
            'Warm ocean conditions': 0.5,
            'Harmful algal bloom': 0.3
        },
        'Pacific Salmon': {
            'Warm ocean conditions': 0.5,
            'Harmful algal bloom': 0.3
        },
        'Anchovy': {
            'Warm ocean conditions': 0.4,
            'Harmful algal bloom': 0.2,
            'Cold water event': 0.3
        },
        'Sardine': {
            'Warm ocean conditions': 0.4,
            'Harmful algal bloom': 0.2,
            'Cold water event': 0.3
        },
        'Albacore Tuna': {
            'Warm ocean conditions': 0.5,
            'Harmful algal bloom': 0.3
        }
    }

    # ========== GULF OF MEXICO ==========
    GULF_OF_MEXICO_SPECIES_VALUE = {
        'Brown Shrimp': 300,
        'White Shrimp': 250,
        'Red Snapper': 200,
        'Oyster': 150,
        'Stone Crab': 100
    }
    
    GULF_OF_MEXICO_SPECIES_DISASTERS = {
        'Brown Shrimp': ['Hurricane', 'Oil spill', 'Flooding'],
        'White Shrimp': ['Hurricane', 'Oil spill', 'Flooding'],
        'Red Snapper': ['Hurricane', 'Oil spill'],
        'Oyster': ['Hurricane', 'Oil spill', 'Flooding'],
        'Stone Crab': ['Hurricane', 'Oil spill', 'Flooding']
    }
    
    GULF_OF_MEXICO_VULNERABILITY = {
        'Brown Shrimp': {
            'Hurricane': 0.3,
            'Oil spill': 0.4,
            'Flooding': 0.2
        },
        'White Shrimp': {
            'Hurricane': 0.3,
            'Oil spill': 0.4,
            'Flooding': 0.2
        },
        'Red Snapper': {
            'Hurricane': 0.3,
            'Oil spill': 0.4
        },
        'Oyster': {
            'Hurricane': 0.3,
            'Oil spill': 0.4,
            'Flooding': 0.2
        },
        'Stone Crab': {
            'Hurricane': 0.3,
            'Oil spill': 0.4,
            'Flooding': 0.2
        }
    }

    # ========== NORTHEAST ==========
    NORTHEAST_SPECIES_VALUE = {
        'American Lobster': 600,
        'Atlantic Cod': 200,
        'Sea Scallop': 300,
        'Atlantic Herring': 150,
        'Haddock': 100
    }
    
    NORTHEAST_SPECIES_DISASTERS = {
        'American Lobster': ['Warm ocean conditions', 'Hurricane', 'Cold water event'],
        'Atlantic Cod': ['Warm ocean conditions', 'Hurricane', 'Cold water event'], 
        'Sea Scallop': ['Warm ocean conditions', 'Hurricane', 'Cold water event', 'Harmful algal bloom'],
        'Atlantic Herring': ['Warm ocean conditions', 'Hurricane', 'Cold water event'],
        'Haddock': ['Warm ocean conditions', 'Hurricane', 'Cold water event']
    }
    
    NORTHEAST_VULNERABILITY = {
        'American Lobster': {
            'Warm ocean conditions': 0.5,
            'Hurricane': 0.3,
            'Cold water event': 0.2
        },
        'Atlantic Cod': {
            'Warm ocean conditions': 0.4,
            'Hurricane': 0.3,
            'Cold water event': 0.3
        },
        'Sea Scallop': {
            'Warm ocean conditions': 0.4,
            'Hurricane': 0.3,
            'Cold water event': 0.2,
            'Harmful algal bloom': 0.3
        },
        'Atlantic Herring': {
            'Warm ocean conditions': 0.4,
            'Hurricane': 0.3,
            'Cold water event': 0.3
        },
        'Haddock': {
            'Warm ocean conditions': 0.4,
            'Hurricane': 0.3,
            'Cold water event': 0.3
        }
    }

    # ========== SOUTHEAST ==========
    SOUTHEAST_SPECIES_VALUE = {
        'Blue Crab': 200,
        'Red Drum': 150,
        'Spotted Seatrout': 100,
        'Spanish Mackerel': 120,
        'Black Sea Bass': 130
    }
    
    SOUTHEAST_SPECIES_DISASTERS = {
        'Blue Crab': ['Hurricane', 'Flooding'],
        'Red Drum': ['Hurricane'],
        'Spotted Seatrout': ['Hurricane'],
        'Spanish Mackerel': ['Hurricane'],
        'Black Sea Bass': ['Hurricane']
    }
    
    SOUTHEAST_VULNERABILITY = {
        'Blue Crab': {
            'Hurricane': 0.3,
            'Flooding': 0.4
        },
        'Red Drum': {
            'Hurricane': 0.3
        },
        'Spotted Seatrout': {
            'Hurricane': 0.3
        },
        'Spanish Mackerel': {
            'Hurricane': 0.3
        },
        'Black Sea Bass': {
            'Hurricane': 0.3
        }
    }

    # ========== HELPER METHODS ==========
    
    @classmethod
    def get_region_data(cls):
        """Get dictionary of all data organized by region"""
        return {
            'Alaska': {
                'species_value': cls.ALASKA_SPECIES_VALUE,
                'species_disasters': cls.ALASKA_SPECIES_DISASTERS,
                'vulnerability': cls.ALASKA_VULNERABILITY
            },
            'West Coast': {
                'species_value': cls.WEST_COAST_SPECIES_VALUE,
                'species_disasters': cls.WEST_COAST_SPECIES_DISASTERS,
                'vulnerability': cls.WEST_COAST_VULNERABILITY
            },
            'Gulf of Mexico': {
                'species_value': cls.GULF_OF_MEXICO_SPECIES_VALUE,
                'species_disasters': cls.GULF_OF_MEXICO_SPECIES_DISASTERS,
                'vulnerability': cls.GULF_OF_MEXICO_VULNERABILITY
            },
            'Northeast': {
                'species_value': cls.NORTHEAST_SPECIES_VALUE,
                'species_disasters': cls.NORTHEAST_SPECIES_DISASTERS,
                'vulnerability': cls.NORTHEAST_VULNERABILITY
            },
            'Southeast': {
                'species_value': cls.SOUTHEAST_SPECIES_VALUE,
                'species_disasters': cls.SOUTHEAST_SPECIES_DISASTERS,
                'vulnerability': cls.SOUTHEAST_VULNERABILITY
            }
        }
    
    @classmethod
    def get_all_regions(cls):
        """Return list of all regions"""
        return list(cls.SPECIES_BY_REGION.keys())
    
    @classmethod
    def get_species_for_region(cls, region):
        """Get species list for a specific region"""
        return cls.SPECIES_BY_REGION.get(region, [])
    
    @classmethod
    def get_all_species(cls):
        """Get unique list of all species across all regions"""
        all_species = set()
        for species_list in cls.SPECIES_BY_REGION.values():
            all_species.update(species_list)
        return sorted(list(all_species))
    
    @classmethod
    def get_species_value(cls, region, species):
        """Get economic value for a species in a region"""
        region_data = cls.get_region_data().get(region, {})
        return region_data.get('species_value', {}).get(species, 50)
    
    @classmethod
    def get_species_disasters(cls, region, species):
        """Get list of disasters affecting a species in a region"""
        region_data = cls.get_region_data().get(region, {})
        return region_data.get('species_disasters', {}).get(species, [])
    
    @classmethod
    def get_species_vulnerability(cls, region, species):
        """Get vulnerability factors for a species in a region (as dictionary)"""
        region_data = cls.get_region_data().get(region, {})
        return region_data.get('vulnerability', {}).get(species, {})
    
    @classmethod
    def get_vulnerability_for_disaster(cls, region, species, disaster_type):
        """Get vulnerability for a specific disaster type"""
        vuln_dict = cls.get_species_vulnerability(region, species)
        return vuln_dict.get(disaster_type, 0.0)
    
    @classmethod
    def get_total_value_by_region(cls):
        """Calculate total fishery value by region"""
        result = {}
        for region, data in cls.get_region_data().items():
            total = sum(data['species_value'].values())
            result[region] = total
        return result


# Example usage
if __name__ == "__main__":
    print("Testing ReferenceData class:\n")
    
    # Test basic access
    print("1. Available Regions:")
    print(ReferenceData.get_all_regions())
    
    print("\n2. Species in Alaska:")
    print(ReferenceData.get_species_for_region('Alaska'))
    
    print("\n3. Snow Crab in Alaska:")
    print(f"   Value: ${ReferenceData.get_species_value('Alaska', 'Snow Crab')}M")
    print(f"   Disasters: {ReferenceData.get_species_disasters('Alaska', 'Snow Crab')}")
    print(f"   Vulnerabilities: {ReferenceData.get_species_vulnerability('Alaska', 'Snow Crab')}")
    
    print("\n4. Specific vulnerability lookup:")
    print(f"   Snow Crab vulnerability to 'Warm ocean conditions': {ReferenceData.get_vulnerability_for_disaster('Alaska', 'Snow Crab', 'Warm ocean conditions')}")
    print(f"   Snow Crab vulnerability to 'Harmful algal bloom': {ReferenceData.get_vulnerability_for_disaster('Alaska', 'Snow Crab', 'Harmful algal bloom')}")
    
    print("\n5. Southeast species:")
    for species in ReferenceData.get_species_for_region('Southeast'):
        vuln = ReferenceData.get_species_vulnerability('Southeast', species)
        print(f"   {species}: {vuln}")
    
    print("\n6. Total value by region:")
    for region, value in ReferenceData.get_total_value_by_region().items():
        print(f"   {region}: ${value}M")