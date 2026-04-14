import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)

class WebConfig(AppConfig):
    name = 'web'

    def ready(self):
        from .services.metadata_lookup import warm_metadata_cache

        try:
            loaded_count = warm_metadata_cache()
            logger.info('Metadata cache warmed on startup (%d show entries)', loaded_count)
        except Exception:
            logger.exception('Failed to warm metadata cache on startup')
