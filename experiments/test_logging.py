from markets.logger.logging import get_logger
from markets.exception import Markets_Exception
import sys

test_log = get_logger()

test_log.info("my name is joti roy")

def test() :
    try :
        a = 10
        b = 0
        c = a/b
    except Markets_Exception as e :
        raise Markets_Exception(e, sys) from e

test()