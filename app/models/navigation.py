# app/models/navigation.py
from pydantic import BaseModel, Field, field_validator, ConfigDict

class NavigationRequest(BaseModel):
    requete: str = Field(..., min_length=1, max_length=2000)

    @field_validator("requete")
    @classmethod
    def requete_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La requête ne peut pas être vide ou composée uniquement d'espaces.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requete": "Je voudrais voir les tarifs s'il vous plaît."
            }
        }
    )

class NavigationResponse(BaseModel):
    page: str
    modele_utilise: str
    page_trouvee: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": "/tarifs",
                "modele_utilise": "llama3:8b",
                "page_trouvee": True
            }
        }
    )
