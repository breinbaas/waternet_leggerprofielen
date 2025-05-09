from pydantic import BaseModel


class Uitgangspunten(BaseModel):
    river_level: float
    dth: float
    onderhoudsdiepte: float
    ipo: str
    required_sf: float
    kruinbreedte: float
    pl_surface_offset: float
    traffic_load_width: float
    traffic_load_magnitude: float
    schematiseringsfactor: float
    modelfactor: float
