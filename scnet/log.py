import logging

log_file = 'training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

torch_logger = logging.getLogger('torch')
torch_logger.setLevel(logging.WARNING)