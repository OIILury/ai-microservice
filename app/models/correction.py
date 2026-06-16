# app/models/correction.py
from pydantic import BaseModel, Field, field_validator, ConfigDict

class CorrectionRequest(BaseModel):
    texte: str = Field(..., min_length=1, max_length=10000)

    @field_validator("texte")
    @classmethod
    def texte_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide ou composé uniquement d'espaces.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texte": "Je mangeai une pomme hier soir, c'était tres bon."
            }
        }
    )

class CorrectionResponse(BaseModel):
    texte_corrige: str
    modele_utilise: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texte_corrige": "Je mangeais une pomme hier soir, c'était très bon.",
                "modele_utilise": "llama3:8b"
            }
        }
    )
