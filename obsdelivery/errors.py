"""Exception types shared by the semantics service and the delivery library."""

from __future__ import annotations

from typing import Iterable, Mapping, Any


class ValidationError(ValueError):
    """A request is malformed or violates a contract."""


class BatchRejected(Exception):
    """An observation admission batch was rejected atomically."""

    def __init__(self, reasons: Iterable[Mapping[str, Any]]):
        self.reasons = list(reasons)
        super().__init__("observation batch rejected: " + repr(self.reasons))


class ReferenceConflict(Exception):
    """A reference version already exists with different immutable content."""


class ReportConflict(Exception):
    """A report id already exists with different immutable content."""


class ReferenceNotFound(KeyError):
    pass


class ReportNotFound(KeyError):
    pass


class ProductConflict(Exception):
    """A product id already exists with different immutable content."""


class ProductNotFound(KeyError):
    pass


class DeliveryNotFound(KeyError):
    pass


class DeliveryIdConflict(Exception):
    """A successful delivery id is being reused with a different request."""


class DeliveryRejected(Exception):
    """No legal delivery exists for at least one requested record.

    A rejected task publishes no delivery rows.  ``attempt_id`` identifies the
    audit entry (if one was recorded), not a successful delivery identity.
    """

    def __init__(
        self,
        delivery_id: str,
        reasons: Iterable[Mapping[str, Any]],
        attempt_id: int | None = None,
    ):
        self.delivery_id = delivery_id
        self.reasons = list(reasons)
        self.attempt_id = attempt_id
        super().__init__(
            f"delivery {delivery_id!r} rejected: " + repr(self.reasons)
        )


class IntegrityError(Exception):
    """Stored immutable content failed independent read-back verification."""


class RecipeConflict(Exception):
    """A splitter recipe id already exists with different immutable content."""


class RecipeNotFound(KeyError):
    pass


class SplitterTaskConflict(Exception):
    """A successful splitter task id is being reused with a different request."""


class SplitterTaskNotFound(KeyError):
    pass


class SplitterRejected(Exception):
    """A splitter task cannot be certified for automatic sorting.

    A non-certifiable channel with a non-zero net coefficient makes the
    total unbounded, so the whole task is refused: no task row is written
    and the failure is only audited in ``splitter_attempts``.  The same
    task id may be retried after the underlying product is corrected.
    """

    def __init__(
        self,
        task_id: str,
        reasons: Iterable[Mapping[str, Any]],
        attempt_id: int | None = None,
    ):
        self.task_id = task_id
        self.reasons = list(reasons)
        self.attempt_id = attempt_id
        super().__init__(
            f"splitter task {task_id!r} rejected: " + repr(self.reasons)
        )
