from markets.exception import Markets_Exception
import sys

try:
    a = 1
    b = 0
    c = a / b
except Exception as e:
    # create Markets_Exception to log the error, then re-raise original
    Markets_Exception(e, sys)
    raise
