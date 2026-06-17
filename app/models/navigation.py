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
    reponse: str           # le texte de réponse généré par le LLM (sans la ligne LIEN)
    page: str | None       # l'URL extraite si présente, sinon None
    modele_utilise: str
    page_trouvee: bool      # True si une URL a été extraite et validée

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reponse": "Nos horaires sont de 9h à 18h en semaine.",
                "page": "/contact",
                "modele_utilise": "llama3:8b",
                "page_trouvee": True
            }
        }
    )
