"""Tests for query optimizer classification heuristics."""

import pytest

from app.pipelines.query_optimizer import QueryOptimizer


@pytest.mark.asyncio
async def test_class_a_reservist_evening_training_flags():
    optimizer = QueryOptimizer()
    query = (
        "I'm a Class A reservist. What narrative should I use to be allowed a meal when I show up "
        "at 1700 in my home unit for training on a Tuesday night and I'll leave around 2200?"
    )

    classification = await optimizer.classify_query(query)

    assert classification.is_class_a_context is True
    assert classification.irregular_hours is True
    assert classification.ordered_outside_normal_hours is False
    assert classification.on_td_or_tasking is False
    assert classification.missed_meal_on_tasking is False
    assert classification.entitlement_likely_denied is True


@pytest.mark.asyncio
async def test_td_evening_briefing_detects_orders_and_missed_meal():
    optimizer = QueryOptimizer()
    query = (
        "I'm on TD in Ottawa and was ordered to attend an evening briefing from 1900 to 2130. "
        "I missed dinner because of the tasking. Can I claim a meal allowance?"
    )

    classification = await optimizer.classify_query(query)

    assert classification.is_class_a_context is False
    assert classification.irregular_hours is True
    assert classification.ordered_outside_normal_hours is True
    assert classification.on_td_or_tasking is True
    assert classification.missed_meal_on_tasking is True
    assert classification.entitlement_likely_denied is False


@pytest.mark.asyncio
async def test_weekend_training_counts_as_tasking():
    optimizer = QueryOptimizer()
    query = (
        "Class A reservist attending weekend exercise at home unit from Saturday 0800 until Sunday 1200, "
        "including overnight accommodation. Are meal and incidental allowances authorized?"
    )

    classification = await optimizer.classify_query(query)

    assert classification.is_class_a_context is True
    assert classification.irregular_hours is True
    assert classification.on_td_or_tasking is True
    assert classification.entitlement_likely_denied is False
