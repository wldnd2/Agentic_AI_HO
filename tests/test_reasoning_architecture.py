from leo_mappo.planner import CHOPlanner, ReasoningResult as PlannerReasoningResult
from leo_mappo.reasoning import LookaheadReasoner, ReasoningResult
from leo_mappo.world_model import GeometryWorldModel


def test_reasoner_exposes_world_model_and_planner() -> None:
    reasoner = LookaheadReasoner({}, {})
    assert isinstance(reasoner.world_model, GeometryWorldModel)
    assert isinstance(reasoner.planner, CHOPlanner)
    assert reasoner.planner.world_model is reasoner.world_model


def test_reasoning_result_remains_reexported() -> None:
    assert ReasoningResult is PlannerReasoningResult
