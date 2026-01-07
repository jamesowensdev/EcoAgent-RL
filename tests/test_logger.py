import logging
import os
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logger

def test_setup_logger_logic_flow():

    unique_name = "VERIFY_LOGIC_LOGGER"

    with patch("os.makedirs") as mock_makedirs, \
         patch("logging.FileHandler") as mock_file_handler:

        with patch.object(logging.Logger, 'hasHandlers', return_value=False):
            setup_logger(unique_name, log_to_file=True)
            mock_makedirs.assert_called_with("logs", exist_ok=True)

            assert mock_file_handler.called

def test_logger_singleton_behavior():

    unique_name = "SINGLETON_CHECK_LOGGER"

    with patch.object(logging.Logger, 'hasHandlers', return_value=True), \
         patch("logging.StreamHandler") as mock_stream:

        setup_logger(unique_name, log_to_file=False)

        assert not mock_stream.called
