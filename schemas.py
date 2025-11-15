"""
Database Schemas for the Pharmacy EdTech App

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name (e.g., Drug -> "drug").
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class Drug(BaseModel):
    """Basic drug information"""
    name: str = Field(..., description="Generic name of the drug")
    brand_names: Optional[List[str]] = Field(default_factory=list, description="Brand names")
    class_name: Optional[str] = Field(None, description="Pharmacological class")
    indications: Optional[List[str]] = Field(default_factory=list, description="Common indications")
    contraindications: Optional[List[str]] = Field(default_factory=list, description="Contraindications")
    side_effects: Optional[List[str]] = Field(default_factory=list, description="Common side effects")
    mechanisms: Optional[str] = Field(None, description="Mechanism of action")

class InteractionRule(BaseModel):
    """Drug-drug interaction rule"""
    drug_a: str = Field(..., description="First drug (by generic name)")
    drug_b: str = Field(..., description="Second drug (by generic name)")
    severity: str = Field(..., description="none | minor | moderate | major | contraindicated")
    description: str = Field(..., description="Description of the interaction and rationale")
    management: Optional[str] = Field(None, description="Suggested management/monitoring")

class QuizQuestion(BaseModel):
    """Quiz question document"""
    topic: str = Field(..., description="Topic or chapter")
    question: str = Field(..., description="Question text")
    options: List[str] = Field(..., description="Multiple choice options")
    answer_index: int = Field(..., ge=0, description="Index of correct option")
    explanation: Optional[str] = Field(None, description="Why the answer is correct")
    difficulty: Optional[str] = Field(None, description="easy | medium | hard")

class PaperSummary(BaseModel):
    """Stored literature summaries"""
    query: str = Field(..., description="Original query")
    title: str = Field(..., description="Paper title")
    url: Optional[str] = Field(None, description="Link to paper")
    abstract: Optional[str] = Field(None, description="Abstract text if available")
    summary: Optional[str] = Field(None, description="AI generated summary")
