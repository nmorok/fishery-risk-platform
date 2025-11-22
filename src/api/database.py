from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Disaster(Base):
    __tablename__ = 'disasters'
    
    disaster_id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    region = Column(String(50), nullable=False)
    fishery = Column(String(100), nullable=False)
    species = Column(String(50), nullable=False)
    disaster_type = Column(String(50), nullable=False)
    duration_years = Column(Integer, nullable=False)
    loss_usd = Column(Float, nullable=False)
    
    # Relationship
    heatwave = relationship("Heatwave", back_populates="disaster", uselist=False)

class Heatwave(Base):
    __tablename__ = 'heatwaves'
    
    heatwave_id = Column(Integer, primary_key=True)
    disaster_id = Column(Integer, ForeignKey('disasters.disaster_id'))
    cumulative_intensity = Column(Float)
    peak_intensity = Column(Float)
    duration_days = Column(Integer)
    max_extent_km2 = Column(Float)
    
    # Relationship
    disaster = relationship("Disaster", back_populates="heatwave")

class SpeciesRegion(Base):
    __tablename__ = 'species_regions'
    
    id = Column(Integer, primary_key=True)
    region = Column(String(50), nullable=False)
    species = Column(String(50), nullable=False)
    annual_value_usd = Column(Float)
    vulnerability_factor = Column(Float)

    def initialize_database(db_path='data/fishery_disasters.db'):
    """
    Create SQLite database and load synthetic data
    """
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    
    # Load synthetic data
    disasters_df = pd.read_csv('data/synthetic/disasters.csv')
    heatwaves_df = pd.read_csv('data/synthetic/heatwaves.csv')
    species_regions_df = pd.read_csv('data/synthetic/species_regions.csv')
    
    # Write to database
    disasters_df.to_sql('disasters', engine, if_exists='replace', index=False)
    heatwaves_df.to_sql('heatwaves', engine, if_exists='replace', index=False)
    species_regions_df.to_sql('species_regions', engine, if_exists='replace', index=False)