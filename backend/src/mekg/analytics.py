from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .models import ValidationStatus
from .parsers import stable_id
from .repository import MEKGRepository, _now


OPPOSITES = [
    ({"increase", "increases", "увеличивает", "повышает", "рост"}, {"decrease", "decreases", "снижает", "уменьшает", "снижение"}),
    ({"applicable", "подходит", "применим"}, {"not applicable", "не подходит", "неприменим"}),
    ({"optimal", "оптималь"}, {"ineffective", "неэффектив"}),
]


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w-]+", value.casefold()))


def _opposed(a: str, b: str) -> bool:
    left, right = a.casefold(), b.casefold()
    for positive, negative in OPPOSITES:
        if (any(token in left for token in positive) and any(token in right for token in negative)) or (
            any(token in right for token in positive) and any(token in left for token in negative)
        ):
            return True
    return False


class MEKGAnalytics:
    def __init__(self, repository: MEKGRepository) -> None:
        self.repository = repository

    def run(self) -> dict[str, int]:
        result = {
            "contradictions": self._contradictions(),
            "consensus": self._consensus(),
            "similar_cases": self._similar_cases(),
            "risks": self._risks(),
            "applicability": self._applicability(),
            "expertise_scores": self._expertise_scores(),
            "knowledge_gaps": self._knowledge_gaps(),
        }
        return result

    def _contradictions(self) -> int:
        rows = self.repository.query(
            """
            MATCH (a:MEKG:Claim)-[:GENERALIZES]->(topic:MEKG)<-[:GENERALIZES]-(b:MEKG:Claim)
            WHERE a.id < b.id AND a.validation_status IN ['machine_validated','expert_validated']
              AND b.validation_status IN ['machine_validated','expert_validated']
            RETURN DISTINCT a.id AS a_id, a.text AS a_text, b.id AS b_id, b.text AS b_text, topic.id AS topic_id
            """
        )
        created = 0
        for row in rows:
            if not _opposed(row["a_text"] or "", row["b_text"] or ""):
                continue
            contradiction_id = stable_id("contradiction", f"{row['a_id']}:{row['b_id']}")
            self.repository.write(
                """
                MATCH (a:Claim {id:$a_id}),(b:Claim {id:$b_id})
                MERGE (c:MEKG:Contradiction {id:$id})
                SET c.type='opposite_conclusion', c.reason='Opposing claims for a shared graph topic',
                    c.severity='medium', c.status='unresolved', c.confidence=0.65,
                    c.validation_status='machine_validated', c.updated_at=$now
                MERGE (c)-[:INVOLVES]->(a)
                MERGE (c)-[:INVOLVES]->(b)
                """,
                {"a_id": row["a_id"], "b_id": row["b_id"], "id": contradiction_id, "now": _now()},
            )
            created += 1
        return created

    def _consensus(self) -> int:
        groups = self.repository.query(
            """
            MATCH (claim:MEKG:Claim)-[:GENERALIZES]->(topic:MEKG)
            WHERE claim.validation_status IN ['machine_validated','expert_validated']
            WITH topic, collect(DISTINCT claim) AS claims, count(DISTINCT claim) AS count
            WHERE count >= 2
            RETURN topic.id AS topic_id, topic.canonical_name AS topic, [c IN claims | c.id] AS claim_ids, count
            """
        )
        for row in groups:
            consensus_id = stable_id("consensus", row["topic_id"])
            self.repository.write(
                """
                MATCH (topic:MEKG {id:$topic_id})
                MERGE (c:MEKG:Consensus {id:$id})
                SET c.topic=$topic, c.supporting_claims=$count, c.confidence=0.7,
                    c.validation_status='machine_validated', c.updated_at=$now
                MERGE (c)-[:SUMMARIZES]->(topic)
                WITH c
                UNWIND $claim_ids AS claim_id
                MATCH (claim:MEKG:Claim {id:claim_id})
                MERGE (c)-[:INVOLVES]->(claim)
                """,
                {"topic_id": row["topic_id"], "id": consensus_id, "topic": row["topic"] or row["topic_id"], "count": row["count"], "claim_ids": row["claim_ids"], "now": _now()},
            )
        return len(groups)

    def _similar_cases(self) -> int:
        rows = self.repository.query(
            """
            MATCH (a:MEKG:Experiment)-[:STUDIES_PROCESS|USES_MATERIAL]->(shared:MEKG)<-[:STUDIES_PROCESS|USES_MATERIAL]-(b:MEKG:Experiment)
            WHERE a.id < b.id
            WITH a,b,collect(DISTINCT shared.id) AS shared_ids
            RETURN a.id AS a_id,b.id AS b_id,shared_ids LIMIT 1000
            """
        )
        for row in rows:
            item_id = stable_id("similar", f"{row['a_id']}:{row['b_id']}")
            self.repository.write(
                """
                MATCH (a:Experiment {id:$a_id}),(b:Experiment {id:$b_id})
                MERGE (s:MEKG:SimilarCase {id:$id})
                SET s.shared_entity_ids=$shared, s.score=0.7, s.validation_status='machine_validated'
                MERGE (s)-[:INVOLVES]->(a) MERGE (s)-[:INVOLVES]->(b)
                """,
                {"a_id": row["a_id"], "b_id": row["b_id"], "id": item_id, "shared": row["shared_ids"]},
            )
        return len(rows)

    def _risks(self) -> int:
        claims = self.repository.query(
            "MATCH (c:MEKG:Claim) WHERE size(coalesce(c.limitations,[]))>0 RETURN c.id AS id,c.limitations AS limitations"
        )
        created = 0
        for claim in claims:
            for limitation in claim["limitations"]:
                risk_id = stable_id("risk", f"{claim['id']}:{limitation.casefold()}")
                self.repository.write(
                    """
                    MATCH (c:Claim {id:$claim_id})
                    MERGE (r:MEKG:Risk {id:$id})
                    SET r.description=$description, r.severity='unknown', r.validation_status='machine_validated'
                    MERGE (r)-[:DERIVED_FROM]->(c)
                    """,
                    {"claim_id": claim["id"], "id": risk_id, "description": limitation},
                )
                created += 1
        return created

    def _applicability(self) -> int:
        rows = self.repository.query(
            "MATCH (t:MEKG:Technology)<-[:GENERALIZES]-(c:MEKG:Claim) RETURN t.id AS technology_id,c.id AS claim_id,c.geo_scope AS geo_scope"
        )
        for row in rows:
            item_id = stable_id("applicability", f"{row['technology_id']}:{row['claim_id']}")
            self.repository.write(
                """
                MATCH (t:Technology {id:$technology_id})
                MATCH (c:Claim {id:$claim_id})
                MERGE (a:MEKG:ApplicabilityAssessment {id:$id})
                SET a.geo_scope=$geo_scope, a.status='evidence_available', a.validation_status='machine_validated'
                MERGE (a)-[:APPLIES_TO]->(t) MERGE (a)-[:DERIVED_FROM]->(c)
                """,
                {"technology_id": row["technology_id"], "claim_id": row["claim_id"], "id": item_id, "geo_scope": row["geo_scope"]},
            )
        return len(rows)

    def _expertise_scores(self) -> int:
        experts = self.repository.query(
            """
            MATCH (e:MEKG:Expert)
            OPTIONAL MATCH (e)-[:EVIDENCED_BY]->(ev)
            WITH e,count(DISTINCT ev) AS evidence_count
            RETURN e.id AS id,e.canonical_name AS name,e.topics AS topics,evidence_count
            """
        )
        created = 0
        for expert in experts:
            topics = expert["topics"] or ["general"]
            for topic in topics:
                score_id = stable_id("expertise", f"{expert['id']}:{str(topic).casefold()}")
                topic_id = stable_id("entity", f"TopicTag:{str(topic).casefold()}")
                score = min(1.0, 0.2 + 0.1 * expert["evidence_count"])
                self.repository.write(
                    """
                    MATCH (e:Expert {id:$expert_id})
                    MERGE (topic:MEKG:CanonicalEntity:TopicTag {id:$topic_id}) SET topic.canonical_name=$topic
                    MERGE (s:MEKG:ExpertiseScore {id:$score_id})
                    SET s.score=$score,s.evidence_count=$evidence_count,s.validation_status='machine_validated'
                    MERGE (s)-[:FOR_EXPERT]->(e) MERGE (s)-[:IN_TOPIC]->(topic)
                    """,
                    {"expert_id": expert["id"], "topic_id": topic_id, "topic": str(topic), "score_id": score_id, "score": score, "evidence_count": expert["evidence_count"]},
                )
                created += 1
        return created

    def _knowledge_gaps(self) -> int:
        patterns = [
            ("mine_water_injection", "закачка шахтных вод", ["закач", "injection"]),
            ("catholyte_flow", "скорость циркуляции католита", ["католит", "catholyte"]),
            ("cold_heap_leaching", "кучное выщелачивание в холодном климате", ["кучн", "heap leaching"]),
            ("water_desalination", "обессоливание шахтных вод", ["обессол", "desalination"]),
        ]
        claims = self.repository.query("MATCH (c:MEKG:Claim) RETURN toLower(c.text) AS text")
        all_text = "\n".join(row["text"] or "" for row in claims)
        created = 0
        for key, description, needles in patterns:
            if any(needle in all_text for needle in needles):
                continue
            gap_id = stable_id("gap", key)
            self.repository.write(
                """
                MERGE (g:MEKG:KnowledgeGap {id:$id})
                SET g.type='missing_evidence',g.description=$description,g.severity='high',
                    g.detected_by='registered_graph_pattern',g.validation_status='machine_validated',g.updated_at=$now
                """,
                {"id": gap_id, "description": f"No verified evidence found for: {description}", "now": _now()},
            )
            created += 1
        return created
