import logging
import os
import sys
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class WebConfig(AppConfig):
    name = "web"

    @staticmethod
    def _is_runserver_parent_process() -> bool:
        # Django autoreload starts a parent and a child process.
        # Only run startup warmups in the child to avoid duplicate work.
        return "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true"

    def ready(self):
        if self._is_runserver_parent_process():
            return

        from .services.embedding_utils import preload_embedding_model
        from .services.metadata_lookup import warm_metadata_cache

        try:
            loaded_count = warm_metadata_cache()
            # info logs are not visible so use warning
            logger.warning(
                "Metadata cache warmed on startup (%d show entries)", loaded_count
            )
        except Exception:
            logger.exception("Failed to warm metadata cache on startup")

        if preload_embedding_model():
            logger.warning("Embedding model loaded on startup")
        else:
            logger.exception("Failed to load embedding model on startup")
