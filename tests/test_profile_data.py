"""Regression checks for recruiter-facing profile facts."""

from pathlib import Path


PROFILE = Path("data/profile.md").read_text(encoding="utf-8")
NORMALIZED_PROFILE = " ".join(PROFILE.split())


def test_profile_contains_verified_positioning_and_metrics():
    assert "Senior Backend Engineer" in NORMALIZED_PROFILE
    assert "100K+ sustained events/sec" in NORMALIZED_PROFILE
    assert "reducing streaming costs by 80%" in NORMALIZED_PROFILE
    assert "seven-figure user platform" in NORMALIZED_PROFILE
    assert "10+ production microservices" in NORMALIZED_PROFILE
    assert "coverage from 30% to 80%" in NORMALIZED_PROFILE


def test_profile_describes_the_kafka_work_accurately():
    assert "did not introduce Kafka as a replacement for RabbitMQ" in NORMALIZED_PROFILE
    assert "publishing directly to Kafka" in NORMALIZED_PROFILE
    assert "removing RabbitMQ from the critical path" in NORMALIZED_PROFILE
    assert "Confluent Cloud to Amazon MSK" in NORMALIZED_PROFILE
    assert "migration of the event architecture from RabbitMQ to Kafka" not in NORMALIZED_PROFILE


def test_profile_separates_personal_project_from_production_work():
    assert "This public portfolio demo is a personal project" in NORMALIZED_PROFILE
    assert "It is separate from the production" in NORMALIZED_PROFILE


def test_profile_has_current_dates_and_backend_target():
    assert "July 2025 - July 2026" in NORMALIZED_PROFILE
    assert "Jul 2025 - present" not in NORMALIZED_PROFILE
    assert "moving toward AI/LLM engineering" not in NORMALIZED_PROFILE
    assert "Senior Backend Engineer roles" in NORMALIZED_PROFILE
