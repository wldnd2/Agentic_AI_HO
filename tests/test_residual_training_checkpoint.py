from __future__ import annotations

from pathlib import Path

from scripts.train_residual_world_model import (
    ValidationCheckpointState,
    _update_best_checkpoint,
)


class _FakeModel:
    def __init__(self) -> None:
        self.current_epoch = 0
        self.saved_epochs: list[int] = []

    def save(self, path: str | Path) -> None:
        assert Path(path).name == "model.npz"
        self.saved_epochs.append(self.current_epoch)


def _apply_losses(
    losses: list[float], patience: int
) -> tuple[ValidationCheckpointState, _FakeModel, int | None]:
    state = ValidationCheckpointState(patience=patience)
    model = _FakeModel()
    stopping_epoch = None
    for epoch, loss in enumerate(losses, start=1):
        model.current_epoch = epoch
        if _update_best_checkpoint(
            state,
            model,
            Path("model.npz"),
            epoch,
            loss,
        ):
            stopping_epoch = epoch
            break
    return state, model, stopping_epoch


def test_middle_epoch_is_retained_as_best_checkpoint() -> None:
    state, model, stopping_epoch = _apply_losses([3.0, 1.0, 1.0, 2.0], patience=0)
    assert stopping_epoch is None
    assert state.best_epoch == 2
    assert state.best_validation_mse == 1.0
    assert model.saved_epochs == [1, 2]
    assert model.saved_epochs[-1] == state.best_epoch


def test_zero_patience_disables_early_stopping() -> None:
    state, model, stopping_epoch = _apply_losses(
        [1.0, 2.0, 3.0, 4.0, 5.0], patience=0
    )
    assert stopping_epoch is None
    assert state.non_improving_epochs == 4
    assert model.saved_epochs == [1]


def test_positive_patience_stops_after_requested_non_improving_epochs() -> None:
    state, model, stopping_epoch = _apply_losses(
        [1.0, 2.0, 3.0, 0.5], patience=2
    )
    assert stopping_epoch == 3
    assert state.non_improving_epochs == 2
    assert state.best_epoch == 1
    assert model.saved_epochs == [1]
