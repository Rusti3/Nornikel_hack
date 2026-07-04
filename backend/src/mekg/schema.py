from __future__ import annotations


ENTITY_LABELS = {
    "Material", "Substance", "ChemicalCompound", "Element", "Ion", "Solution", "Ore", "Concentrate",
    "Slag", "Matte", "Cathode", "Anode", "Waste", "GasStream", "WaterStream", "Product", "ByProduct",
    "Phase", "GasComponent", "Process", "ProcessStep", "Technology", "Method", "Regime", "UnitOperation",
    "HydrometallurgicalProcess", "PyrometallurgicalProcess", "EnvironmentalProcess", "WasteProcessingProcess",
    "ElectrowinningProcess", "LeachingProcess", "SmeltingProcess", "DesalinationProcess", "GasCleaningProcess",
    "MineWaterInjectionProcess", "Equipment", "Facility", "Cell", "Furnace", "Reactor", "Filter",
    "MembraneUnit", "Pump", "Pipeline", "GasCleaningSystem", "ElectrowinningBath", "FlashSmeltingFurnace",
    "Parameter", "Property", "Problem", "Expert", "Author", "Team", "Lab", "Organization", "Project",
    "GeoRegion", "Country", "Site", "GeoScope", "TopicTag",
}

FACT_LABELS = {
    "Experiment", "DistributionExperiment", "CaseStudy", "IndustrialPractice", "LabTest", "PilotTest",
    "FullScaleOperation", "Sample", "Feed", "Output", "Condition", "Measurement", "DistributionMeasurement",
    "Effect", "EfficiencyMetric", "EconomicMetric", "EnvironmentalMetric", "Claim", "Conclusion",
    "Recommendation", "Limitation", "Assumption", "Decision", "Hypothesis", "Contradiction", "Consensus",
    "KnowledgeGap", "ResearchDirection", "Risk", "SimilarCase", "ApplicabilityAssessment",
}

SOURCE_LABELS = {"Document", "DocumentVersion", "Publication", "Page", "Slide", "Chunk", "Table", "TableRow", "Figure", "Formula"}

ALLOWED_LABELS = ENTITY_LABELS | FACT_LABELS | SOURCE_LABELS | {
    "EvidencePack", "ValidationRecord", "ClaimVersion", "RelationshipAssertion", "Term", "Alias", "Unit",
    "Dimension", "ExpertiseScore", "GraphPattern", "ClaimGroup", "ConditionDifference", "ExtractionCandidate",
    "OntologyClass", "OntologyProperty", "ConceptScheme",
}

ALLOWED_RELATIONSHIPS = {
    "HAS_VERSION", "HAS_PAGE", "HAS_PUBLICATION", "HAS_CHUNK", "HAS_TABLE", "HAS_ROW", "HAS_FIGURE",
    "HAS_FORMULA", "MENTIONS", "EVIDENCED_BY", "HAS_EVIDENCE", "HAS_COMPONENT", "HAS_PHASE", "CONTAINS",
    "HAS_CONTAMINANT", "HAS_STEP", "USES_MATERIAL", "PRODUCES", "USES_EQUIPMENT", "OPERATES_AT",
    "HAS_REGIME", "APPLIES_TO", "SOLVES", "STUDIES_PROCESS", "USES_FEED", "RUN_ON", "HAS_CONDITION",
    "PRODUCED_MEASUREMENT", "PRODUCED_EFFECT", "PERFORMED_BY", "DESCRIBED_IN", "MEASURES_PROPERTY",
    "HAS_PARAMETER", "HAS_UNIT", "HAS_METHOD", "HAS_SAMPLE", "SUPPORTS", "SUPPORTED_BY", "CONTRADICTS",
    "GENERALIZES", "HAS_LIMITATION", "BASED_ON", "SUPERSEDED_BY", "SUPERSEDES", "VALIDATED_BY", "AUTHOR_OF",
    "MEMBER_OF", "PART_OF", "EXPERT_IN", "WORKED_ON", "LOCATED_IN", "APPLIED_IN", "HAS_GEO_SCOPE",
    "DESCRIBES_PRACTICE_IN", "MISSING_FOR", "MISSING_PROCESS", "MISSING_PROPERTY", "MISSING_CONDITION",
    "MISSING_GEO_REGION", "DERIVED_FROM", "INVOLVES", "EXPLAINED_BY", "SUMMARIZES", "HAS_TERM", "HAS_ALIAS",
    "HAS_DIMENSION", "CONVERTS_TO", "FROM_PHASE", "TO_PHASE", "FOR_ELEMENT", "HAS_DISTRIBUTION_RESULT",
    "MEASURES_DISTRIBUTION_OF", "HAS_INPUT_CONDITION", "HAS_OUTPUT_TARGET", "HAS_PERFORMANCE",
    "HAS_ECONOMIC_METRIC", "VALIDATED_BY_CASE", "NEXT_CHUNK", "FIRST_CHUNK", "HAS_ASSERTION", "ASSERTS_SOURCE",
    "ASSERTS_TARGET", "FOR_EXPERT", "IN_TOPIC", "PARTICIPATED_IN", "NEEDS_EXPERT_REVIEW",
    "SUBCLASS_OF",
}

CANONICAL_PARENT = {
    label: "CanonicalEntity"
    for label in ENTITY_LABELS
}
